"""
Annotate candidate stimuli with Expected Information Gain (EIG) over PyMC models.

Each stimulus has "sequence_a"/"sequence_b" keys. The cognitive models are PyMC
models (module-level `model: pm.Model`); EIG is computed from each model's
**prior-predictive** p_left for the stimulus — no MCMC fit is needed at design
time. Raw stimuli are featurized (via the project's `featurize_stimulus`) into
the numeric columns the models read through `pm.Data`. EIG is added as a float
field; stimuli are sorted by EIG descending on output.

Usage (CLI):
    python3 -m src.pipelines.outer_loop.eig \\
        --candidates candidates.json \\
        --models-dir PATH/cognitive_models \\
        --featurize  PATH/projects/<project>/preprocess.py \\
        --registry   PATH/model_registry.yaml \\
        --out        PATH/design/stimuli.json \\
        --top        20

    # --out defaults to stdout if omitted
    # --top defaults to all stimuli (no truncation)
    # --registry is optional (uniform prior over models if omitted)
    # --featurize is optional (omit if candidates already carry feature columns)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _load_featurizer(
    featurize_path: Optional[Path],
) -> Optional[Callable[[str, str], Dict[str, Any]]]:
    if featurize_path is None:
        return None
    featurize_path = Path(featurize_path)
    if not featurize_path.exists():
        raise FileNotFoundError(f"featurize module not found: {featurize_path}")
    spec = importlib.util.spec_from_file_location("_eig_featurize", featurize_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load featurize module from {featurize_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_eig_featurize"] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "featurize_stimulus", None)
    if fn is None:
        raise AttributeError(f"{featurize_path} has no featurize_stimulus()")
    return fn


def annotate(
    candidates: List[Dict[str, Any]],
    models_dir: Path,
    registry_path: Optional[Path] = None,
    *,
    featurize_path: Optional[Path] = None,
    n_samples: int = 200,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Annotate each candidate with an "eig" key and sort by EIG descending.

    candidates: list of dicts, each with "sequence_a"/"sequence_b" (and possibly
        already-derived feature columns).
    models_dir: cognitive_models/ directory (models_manifest.yaml + PyMC .py files).
    registry_path: optional model_registry.yaml for a weighted model prior.
    featurize_path: optional path to a module exposing featurize_stimulus(seq_a,
        seq_b) -> dict of numeric feature columns. If given, those columns are
        merged into each stimulus before EIG so the PyMC models can read them.
    n_samples: prior-predictive draws per model per stimulus.
    """
    import yaml

    from src.models.pymc_inference import (
        expected_information_gain_prior_pymc,  # type: ignore
    )
    from src.models.theorist.loader import get_model_names_from_manifest  # type: ignore

    models_dir = Path(models_dir)
    manifest_path = models_dir / "models_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"models_manifest.yaml not found at {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    model_names = get_model_names_from_manifest(manifest, models_dir)
    if not model_names:
        raise ValueError(f"No loadable models found in {models_dir}")

    model_weights: Dict[str, float] = {}
    if registry_path and Path(registry_path).exists():
        reg = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8")) or {}
        model_weights = reg.get("theories", {}) or {}

    featurize = _load_featurizer(featurize_path)

    # Screen out models that cannot be evaluated on a bare stimulus — e.g. a
    # carried-forward model with a participant-level pm.Data (participant_id) that
    # stimulus feature rows never carry. One such model would otherwise raise
    # inside prior_predict_p_left and abort the entire annotation. Probe each model
    # against a representative featurized stimulus, drop the unbindable ones
    # loudly, and keep the rest; fail only if none can be evaluated.
    if candidates:
        from src.models.pymc_inference import (  # type: ignore
            load_pymc_model_cached,
            make_stim_data,
        )

        probe = dict(candidates[0])
        if featurize is not None:
            probe.update(
                featurize(candidates[0]["sequence_a"], candidates[0]["sequence_b"])
            )
        probe.setdefault("chose_left", 0)

        usable: List[str] = []
        for name in model_names:
            try:
                make_stim_data(load_pymc_model_cached(name, models_dir), [probe])
            except Exception as e:  # noqa: BLE001 — unbindable model can't be scored
                print(
                    f"  [drop] EIG: model {name!r} cannot be evaluated on a "
                    f"stimulus ({type(e).__name__}: {e}); excluding it from EIG.",
                    flush=True,
                )
                continue
            usable.append(name)
        if not usable:
            raise ValueError(
                f"No models in {models_dir} can be evaluated on a stimulus row "
                "(every model requires columns absent from stimuli, e.g. "
                "participant_id); cannot compute EIG."
            )
        model_names = usable

    results = []
    for item in candidates:
        feature_row: Dict[str, Any] = dict(item)
        if featurize is not None:
            feature_row.update(featurize(item["sequence_a"], item["sequence_b"]))
        # The observed-response container is required as a pm.Data input but its
        # value is ignored for prior-predictive p_left — pass a dummy.
        feature_row.setdefault("chose_left", 0)
        eig_val = expected_information_gain_prior_pymc(
            feature_row,
            model_names,
            models_dir,
            model_weights=model_weights or None,
            n_samples=n_samples,
            seed=seed,
        )
        results.append({**item, "eig": round(eig_val, 6)})

    results.sort(key=lambda x: -x["eig"])
    return results


@dataclass
class Args:
    """Annotate candidate stimuli with EIG over PyMC models and sort descending."""

    candidates: Path
    """JSON file with a list of {sequence_a, sequence_b} dicts."""
    models_dir: Path
    """Path to the cognitive_models/ directory."""
    featurize: Optional[Path] = None
    """Path to a module exposing featurize_stimulus() (e.g. projects/<project>/preprocess.py)."""
    registry: Optional[Path] = None
    """Path to model_registry.yaml (optional; uniform prior if omitted)."""
    out: Optional[Path] = None
    """Output JSON file path (default: stdout)."""
    top: Optional[int] = None
    """Keep only the top N stimuli by EIG."""
    n_samples: int = 200
    """Prior-predictive draws per model per stimulus."""


def main(args: Args) -> None:
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    if not isinstance(candidates, list):
        print("Error: candidates file must contain a JSON list", file=sys.stderr)
        sys.exit(1)

    annotated = annotate(
        candidates=candidates,
        models_dir=args.models_dir,
        registry_path=args.registry,
        featurize_path=args.featurize,
        n_samples=args.n_samples,
    )

    if args.top is not None:
        annotated = annotated[: args.top]

    output = json.dumps(annotated, indent=2)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        eig_vals = [s["eig"] for s in annotated]
        print(
            f"Wrote {len(annotated)} stimuli to {args.out} "
            f"(EIG range: {min(eig_vals):.4f} – {max(eig_vals):.4f})",
            flush=True,
        )
    else:
        print(output)


if __name__ == "__main__":
    main(tyro.cli(Args))
