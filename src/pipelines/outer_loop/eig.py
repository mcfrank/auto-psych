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

Exhaustive design mode (no candidates file): enumerate EVERY sequence pair over
the given lengths, score all of them in one batched prior-predictive pass per
model, and greedily select the set with maximal *joint* EIG about model
identity (src.models.eig_selection):

    python3 -m src.pipelines.outer_loop.eig \\
        --exhaustive --select 32 --lengths 4 5 6 7 8 \\
        --models-dir PATH/cognitive_models \\
        --featurize  PATH/projects/<project>/preprocess.py \\
        --registry   PATH/model_registry.yaml \\
        --out        PATH/design/stimuli.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import tyro
from pyprojroot import here

# Ensure repo root on path so "import src..." works when run as a module/script.
# Must precede the (function-level) src imports below, hence here() rather than
# the canonical src.runtime.config.REPO_ROOT (same resolution).
sys.path.insert(0, str(here()))


def _load_featurizer(
    featurize_path: Optional[Path],
) -> Optional[Callable[[str, str], Dict[str, Any]]]:
    """Return the project's featurize_stimulus, or None if --featurize was omitted."""
    if featurize_path is None:
        return None
    from src.pipelines.outer_loop.featurizer import load_featurizer  # type: ignore

    return load_featurizer(featurize_path)


def _load_model_names(models_dir: Path) -> List[str]:
    from src.models.model_manifest import read_loadable_model_names  # type: ignore

    model_names = read_loadable_model_names(models_dir)
    if not model_names:
        raise ValueError(f"No loadable models found in {models_dir}")
    return model_names


def _load_model_weights(registry_path: Optional[Path]) -> Dict[str, float]:
    if registry_path is None:
        return {}
    from src.registry.io import load_registry  # type: ignore

    path = Path(registry_path)
    if not path.is_file():
        raise FileNotFoundError(f"Explicit model registry does not exist: {path}")
    return dict(load_registry(path)["theories"])


def _screen_usable_models(
    model_names: List[str], models_dir: Path, probe_row: Dict[str, Any]
) -> List[str]:
    """Drop models that cannot be evaluated on a bare stimulus row.

    E.g. a carried-forward model with a participant-level pm.Data
    (participant_id) that stimulus feature rows never carry. One such model
    would otherwise raise inside the prior-predictive pass and abort the entire
    annotation. Probe each model against a representative featurized stimulus,
    drop the unbindable ones loudly, and keep the rest; fail only if none can
    be evaluated.

    This is the ONE place the pipeline is allowed to omit a model from the
    hypothesis set, and only for the data-binding reason above: a model that
    fails because its *code* is broken (``BROKEN_MODEL_CODE_ERRORS``) raises.
    """
    from src.models.pymc_inference import (  # type: ignore
        BROKEN_MODEL_CODE_ERRORS,
        load_pymc_model_cached,
        make_stim_data,
    )

    usable: List[str] = []
    for name in model_names:
        try:
            make_stim_data(load_pymc_model_cached(name, models_dir), [probe_row])
        except BROKEN_MODEL_CODE_ERRORS as e:
            raise RuntimeError(
                f"model {name!r} in {models_dir} is broken "
                f"({type(e).__name__}: {e}). That is a code error, not a "
                "stimulus-binding mismatch — fix the model rather than letting "
                "EIG silently renormalize over the models that happen to load."
            ) from e
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
    return usable


def _feature_row(
    item: Dict[str, Any], featurize: Optional[Callable[[str, str], Dict[str, Any]]]
) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(item)
    if featurize is not None:
        row.update(featurize(item["sequence_a"], item["sequence_b"]))
    # The observed-response container is required as a pm.Data input but its
    # value is ignored for prior-predictive p_left — pass a dummy.
    row.setdefault("chose_left", 0)
    return row


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
    n_samples: prior-predictive draws per model.

    All candidates are scored in one batched prior-predictive pass per model
    (identical values to the historical per-stimulus path with the same seed,
    ~3,500x faster).
    """
    from src.models.pymc_inference import (  # type: ignore
        eig_from_prior_means,
        prior_predict_p_left_batch,
    )

    models_dir = Path(models_dir)
    model_names = _load_model_names(models_dir)
    model_weights = _load_model_weights(registry_path)
    featurize = _load_featurizer(featurize_path)

    if not candidates:
        return []
    rows = [_feature_row(item, featurize) for item in candidates]
    model_names = _screen_usable_models(model_names, models_dir, rows[0])

    means = prior_predict_p_left_batch(
        model_names, models_dir, rows, n_samples=n_samples, seed=seed
    )
    results = []
    for i, item in enumerate(candidates):
        preds = {m: float(means[m][i]) for m in means}
        eig_val = eig_from_prior_means(preds, model_weights or None)
        results.append({**item, "eig": round(eig_val, 6)})

    results.sort(key=lambda x: -x["eig"])
    return results


def _posterior_p_left_draws(
    model_names: List[str],
    models_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    responses_csv: Path,
    fit_cache_dir: Optional[Path],
    max_draws: int,
    seed: int,
    fit_draws: Optional[int] = None,
    fit_tune: Optional[int] = None,
    fit_chains: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-draw posterior-predictive p_left over ``rows`` for each model.

    Fits every model on ``responses_csv`` (design-time MCMC settings from
    ``src.models.mcmc_defaults`` unless overridden) and predicts p_left draws
    for the stimulus pool, thinned to ``max_draws`` posterior samples.
    """
    from src.models.mcmc_defaults import (  # type: ignore
        DESIGN_TWIN_CHAINS,
        DESIGN_TWIN_DRAWS,
        DESIGN_TWIN_TUNE,
    )
    from src.models.pymc_inference import fit_model, make_stim_data  # type: ignore

    draws: Dict[str, Any] = {}
    for name in model_names:
        fitted = fit_model(
            name,
            models_dir,
            responses_csv,
            cache_dir=fit_cache_dir,
            draws=fit_draws if fit_draws is not None else DESIGN_TWIN_DRAWS,
            tune=fit_tune if fit_tune is not None else DESIGN_TWIN_TUNE,
            chains=fit_chains if fit_chains is not None else DESIGN_TWIN_CHAINS,
        )
        stim_data = make_stim_data(fitted.model, rows)
        draws[name] = fitted.predict_p_left_draws(
            stim_data, seed=seed, max_draws=max_draws
        )
    return draws


def design_exhaustive(
    models_dir: Path,
    registry_path: Optional[Path] = None,
    *,
    featurize_path: Optional[Path] = None,
    lengths: tuple = (4, 5, 6, 7, 8),
    n_select: int = 32,
    n_samples: int = 200,
    n_scenarios: int = 1000,
    seed: int = 42,
    responses_csv: Optional[Path] = None,
    fit_cache_dir: Optional[Path] = None,
    fit_draws: Optional[int] = None,
    fit_tune: Optional[int] = None,
    fit_chains: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Select the max-joint-EIG stimulus set from the FULL pair universe.

    Enumerates every distinct sequence pair over ``lengths`` (cross-length
    pairs included), scores all of them in one batched per-draw pass per
    model, and greedily selects the ``n_select`` stimuli with maximal joint
    EIG about model identity. No candidates file — the pool is the whole
    space, so nothing an agent could conjecture is outside it.

    Without ``responses_csv``, per-draw p_left comes from each model's
    **prior** predictive (experiment 1). With ``responses_csv``, each model is
    first fitted on those responses (MCMC, at the design-time settings from
    ``src.models.mcmc_defaults`` unless fit_* override them) and per-draw
    p_left comes from its **posterior** predictive, thinned to ``n_samples``
    draws — sequential design informed by the previous experiment.

    Returns stimuli in selection (greedy) order, each with:
      - "eig": the stimulus's marginal EIG (bits), same meaning as `annotate`;
      - "selection_rank": 1-based greedy pick order;
      - "joint_eig_bits": in-sample joint EIG of the set up to this stimulus.
    """
    from src.models.eig_selection import select_n_joint_eig  # type: ignore
    from src.models.pymc_inference import (  # type: ignore
        eig_from_prior_means,
        prior_predict_p_left_draws,
    )
    from src.subjective_randomness.stimulus_design import (  # type: ignore
        enumerate_all_pairs,
    )

    models_dir = Path(models_dir)
    if responses_csv is not None and not Path(responses_csv).exists():
        raise FileNotFoundError(
            f"Posterior exhaustive design needs responses at {responses_csv}, "
            "but the file is missing."
        )
    model_names = _load_model_names(models_dir)
    model_weights = _load_model_weights(registry_path)
    featurize = _load_featurizer(featurize_path)

    pool = enumerate_all_pairs(list(lengths))
    rows = [_feature_row(item, featurize) for item in pool]
    model_names = _screen_usable_models(model_names, models_dir, rows[0])
    if model_weights and not any(model_weights.get(n, 0.0) > 0 for n in model_names):
        print(
            f"  [design] registry weights over {sorted(model_weights)} do not "
            f"overlap this model set {model_names}; using a uniform model prior.",
            flush=True,
        )
    basis = "prior predictive" if responses_csv is None else "posterior predictive"
    print(
        f"Exhaustive design: {len(pool):,d} pairs over lengths {list(lengths)}, "
        f"{len(model_names)} models ({basis}), selecting {n_select}.",
        flush=True,
    )

    if responses_csv is None:
        draws = prior_predict_p_left_draws(
            model_names, models_dir, rows, n_samples=n_samples, seed=seed
        )
    else:
        draws = _posterior_p_left_draws(
            model_names,
            models_dir,
            rows,
            responses_csv=Path(responses_csv),
            fit_cache_dir=fit_cache_dir,
            max_draws=n_samples,
            seed=seed,
            fit_draws=fit_draws,
            fit_tune=fit_tune,
            fit_chains=fit_chains,
        )
    selection = select_n_joint_eig(
        draws,
        n_select,
        model_weights=model_weights or None,
        n_scenarios=n_scenarios,
        seed=seed,
    )

    means = {m: arr.mean(axis=0) for m, arr in draws.items()}
    results = []
    for rank, (idx, joint_bits) in enumerate(
        zip(selection.indices, selection.joint_eig_bits), start=1
    ):
        preds = {m: float(means[m][idx]) for m in means}
        results.append(
            {
                **pool[idx],
                "eig": round(eig_from_prior_means(preds, model_weights or None), 6),
                "selection_rank": rank,
                "joint_eig_bits": round(joint_bits, 6),
            }
        )
    return results


@dataclass
class Args:
    """Annotate candidate stimuli with EIG over PyMC models and sort descending."""

    models_dir: Path
    """Path to the cognitive_models/ directory."""
    candidates: Optional[Path] = None
    """JSON file with a list of {sequence_a, sequence_b} dicts (legacy mode)."""
    featurize: Optional[Path] = None
    """Path to a module exposing featurize_stimulus() (e.g. projects/<project>/preprocess.py)."""
    registry: Optional[Path] = None
    """Path to model_registry.yaml (optional; uniform prior if omitted)."""
    out: Optional[Path] = None
    """Output JSON file path (default: stdout)."""
    top: Optional[int] = None
    """Keep only the top N stimuli by EIG (legacy mode only)."""
    n_samples: int = 200
    """Prior-predictive draws per model."""
    exhaustive: bool = False
    """Ignore --candidates: enumerate the full pair universe over --lengths and
    select the max-joint-EIG set of --select stimuli."""
    lengths: tuple = (4, 5, 6, 7, 8)
    """Sequence lengths for the exhaustive pair universe."""
    select: Optional[int] = None
    """Stimulus-set size for exhaustive joint-EIG selection (required with --exhaustive)."""
    n_scenarios: int = 1000
    """Monte Carlo scenarios for joint-EIG gain estimation (exhaustive mode)."""
    seed: int = 42
    """Seed for prior draws and selection scenarios."""
    responses: Optional[Path] = None
    """Previous experiment's responses.csv: fit each model on it and design from
    the POSTERIOR predictive instead of the prior (exhaustive mode only)."""
    fit_cache: Optional[Path] = None
    """Cache dir for the design-time MCMC fits (with --responses)."""


def _write_output(stimuli: List[Dict[str, Any]], out: Optional[Path]) -> None:
    output = json.dumps(stimuli, indent=2)
    if out:
        out.write_text(output, encoding="utf-8")
        eig_vals = [s["eig"] for s in stimuli]
        print(
            f"Wrote {len(stimuli)} stimuli to {out} "
            f"(EIG range: {min(eig_vals):.4f} – {max(eig_vals):.4f})",
            flush=True,
        )
    else:
        print(output)


def main(args: Args) -> None:
    if args.exhaustive and args.candidates is not None:
        print(
            "Error: --exhaustive enumerates the whole pair universe; "
            "--candidates must not also be given.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.exhaustive and args.responses is not None:
        print(
            "Error: --responses (posterior design) requires --exhaustive.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.exhaustive:
        if args.select is None:
            print("Error: --exhaustive requires --select N.", file=sys.stderr)
            sys.exit(1)
        selected = design_exhaustive(
            models_dir=args.models_dir,
            registry_path=args.registry,
            featurize_path=args.featurize,
            lengths=tuple(args.lengths),
            n_select=args.select,
            n_samples=args.n_samples,
            n_scenarios=args.n_scenarios,
            seed=args.seed,
            responses_csv=args.responses,
            fit_cache_dir=args.fit_cache,
        )
        _write_output(selected, args.out)
        return

    if args.candidates is None:
        print(
            "Error: --candidates is required (or use --exhaustive --select N).",
            file=sys.stderr,
        )
        sys.exit(1)
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
        seed=args.seed,
    )

    if args.top is not None:
        annotated = annotated[: args.top]
    _write_output(annotated, args.out)


if __name__ == "__main__":
    main(tyro.cli(Args))
