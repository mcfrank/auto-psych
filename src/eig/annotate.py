"""
Annotate a list of candidate stimuli with Expected Information Gain (EIG).

Each stimulus must have "sequence_a" and "sequence_b" keys. EIG is added as a
float field on each item. Stimuli are sorted by EIG descending on output.

Usage (CLI):
    python3 -m src.eig.annotate \\
        --candidates candidates.json \\
        --models-dir PATH/cognitive_models \\
        --registry  PATH/model_registry.yaml \\
        --out       PATH/design/stimuli.json \\
        --top       20

    # --out defaults to stdout if omitted
    # --top defaults to all stimuli (no truncation)
    # --registry is optional (uniform prior if omitted)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


RESPONSE_OPTIONS = ["left", "right"]


def _eig(
    stimulus: tuple,
    model_names: List[str],
    models_dir: Path,
    model_weights: Dict[str, float],
) -> float:
    """
    EIG = H(M) − E_R[H(M|R)] in bits.
    M is the discrete model-identity variable; R ∈ {left, right}.
    model_weights is the prior P(M); uniform over models if empty.
    """
    from src.models.randomness import get_model_predictions  # type: ignore

    # ── likelihood: P(R | M, stimulus) ───────────────────────────────────────
    preds = get_model_predictions(stimulus, RESPONSE_OPTIONS, model_names, models_dir)

    # ── validate all inputs before touching any math ──────────────────────────
    if not preds:
        raise ValueError("No predictions returned.")
    for m, pred in preds.items():
        if any(p < 0 for p in pred.values()):
            raise ValueError(f"Model {m!r} returned a negative probability: {pred}")
        total = sum(pred.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Model {m!r} predictions sum to {total:.6f}, expected 1")
    if any(w < 0 for w in model_weights.values()):
        raise ValueError(f"Negative model weight in: {model_weights}")
    raw_weights = {m: model_weights.get(m, 0.0) for m in preds} if model_weights else {}
    if raw_weights and sum(raw_weights.values()) <= 0:
        raise ValueError(f"Model weights (for models in preds) sum to 0; at least one must be positive")

    # ── prior: P(M) ──────────────────────────────────────────────────────────
    if raw_weights:
        total_w = sum(raw_weights.values())
        p_model = {m: raw_weights[m] / total_w for m in preds}
    else:
        p_model = {m: 1.0 / len(preds) for m in preds}

    # ── marginal: P(R) = Σ_m P(R | M=m) · P(M=m) ───────────────────────────
    p_left = sum(preds[m]["left"] * p_model[m] for m in preds)
    p_right = 1.0 - p_left

    # Degenerate stimulus: all models agree on the response, so observing it
    # gives no information about model identity → EIG = 0
    if p_left == 0.0 or p_right == 0.0:
        return 0.0

    # ── prior entropy: H(M) ──────────────────────────────────────────────────
    # 0·log₂(0) ≡ 0 by convention; skip those terms
    h_prior = -sum(p * math.log2(p) for p in p_model.values() if p > 0)

    # ── posterior entropy: H(M | R=r) ────────────────────────────────────────
    # P(M | R=r) ∝ P(R=r | M) · P(M), normalised by the marginal P(R=r)
    def h_posterior_given(response: str, p_r: float) -> float:
        p_m_given_r = [preds[m][response] * p_model[m] / p_r for m in preds]
        return -sum(p * math.log2(p) for p in p_m_given_r if p > 0)

    # ── EIG = H(M) − E_R[H(M|R)] ────────────────────────────────────────────
    e_h_posterior = (p_left  * h_posterior_given("left",  p_left) +
                     p_right * h_posterior_given("right", p_right))
    return max(0.0, h_prior - e_h_posterior)


def annotate(
    candidates: List[Dict[str, Any]],
    models_dir: Path,
    registry_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Annotate each candidate dict with an "eig" key and sort by EIG descending.

    candidates: list of dicts, each with "sequence_a" and "sequence_b"
    models_dir: path to cognitive_models/ directory (contains models_manifest.yaml + .py files)
    registry_path: optional path to model_registry.yaml for weighted prior

    Returns a new list of dicts with "eig" added, sorted descending.
    """
    from src.models.loader import get_model_names_from_manifest  # type: ignore
    import yaml

    manifest_path = models_dir / "models_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"models_manifest.yaml not found at {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    model_names = get_model_names_from_manifest(manifest, models_dir)
    if not model_names:
        raise ValueError(f"No loadable models found in {models_dir}")

    model_weights: Dict[str, float] = {}
    if registry_path and registry_path.exists():
        reg = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        model_weights = reg.get("theories", {})

    results = []
    for item in candidates:
        stimulus = (item["sequence_a"], item["sequence_b"])
        eig_val = _eig(stimulus, model_names, models_dir, model_weights)
        results.append({**item, "eig": round(eig_val, 6)})

    results.sort(key=lambda x: -x["eig"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate candidate stimuli with EIG and sort descending"
    )
    parser.add_argument(
        "--candidates", required=True,
        help="JSON file with list of {sequence_a, sequence_b} dicts",
    )
    parser.add_argument(
        "--models-dir", required=True,
        help="Path to cognitive_models/ directory",
    )
    parser.add_argument(
        "--registry", default=None,
        help="Path to model_registry.yaml (optional; uniform prior if omitted)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Keep only the top N stimuli by EIG (default: all)",
    )
    args = parser.parse_args()

    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    if not isinstance(candidates, list):
        print("Error: candidates file must contain a JSON list", file=sys.stderr)
        sys.exit(1)

    annotated = annotate(
        candidates=candidates,
        models_dir=Path(args.models_dir),
        registry_path=Path(args.registry) if args.registry else None,
    )

    if args.top is not None:
        annotated = annotated[: args.top]

    output = json.dumps(annotated, indent=2)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        eig_vals = [s["eig"] for s in annotated]
        print(
            f"Wrote {len(annotated)} stimuli to {args.out} "
            f"(EIG range: {min(eig_vals):.4f} – {max(eig_vals):.4f})",
            flush=True,
        )
    else:
        print(output)


if __name__ == "__main__":
    main()
