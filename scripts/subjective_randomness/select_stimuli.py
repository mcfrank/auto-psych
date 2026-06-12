"""CLI: select the most model-discriminating stimuli from a candidate pool.

Scores each candidate sequence pair by the expected information (bits) it
carries about *which* reference model generated the response — the mutual
information between model identity and the binary choice (see
``src/subjective_randomness/stimulus_design.py``). Writes the ranked (optionally
top-``k``) stimuli, annotated with ``discrimination_eig``, as a stimuli JSON
ready to feed back into recovery.

Use this to fix poor model recoverability: a handful of discriminating items
separates near-tied models far better than many undiagnostic ones. This is the
fast, MCMC-free design pass; ``src/pipelines/outer_loop/eig.py`` is the full
PyMC prior-predictive version.

Usage:
    uv run python scripts/subjective_randomness/select_stimuli.py \\
        --candidates pool.json \\
        --out data/subjective_randomness/discriminating_stimuli.json \\
        --top 20
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402
from src.subjective_randomness.stimulus_design import (  # noqa: E402
    default_model_family_names,
    family_predict_fns,
    rank_stimuli,
)


@dataclass
class Args:
    """Select model-discriminating stimuli from a candidate pool."""

    candidates: Path
    """JSON list of candidate stimuli ({"sequence_a", "sequence_b"})."""
    out: Path
    """Output stimuli JSON (ranked, annotated with discrimination_eig)."""
    top: Optional[int] = None
    """Keep only the top-N most discriminating stimuli (default: keep all, ranked)."""
    models: Optional[List[str]] = None
    """Model family names to discriminate (default: all reference families)."""
    param_samples: Optional[int] = None
    """If set, average p_left over N parameter draws (prior-predictive) per model."""
    seed: int = 0
    """RNG seed for the prior-predictive parameter draws."""


def main(args: Args) -> None:
    candidates_path = resolve_path(args.candidates)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not candidates:
        raise ValueError(f"No candidate stimuli in {candidates_path}.")

    model_names = args.models or default_model_family_names()
    predict_fns = family_predict_fns(
        model_names, param_samples=args.param_samples, seed=args.seed
    )
    ranked = rank_stimuli(candidates, predict_fns)
    selected = ranked[: args.top] if args.top is not None else ranked

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    eigs = [s["discrimination_eig"] for s in selected]
    print(
        f"Scored {len(candidates)} candidates against {len(model_names)} models "
        f"{model_names}; kept {len(selected)}."
    )
    print(
        f"  discrimination_eig (bits): max={max(eigs):.3f} "
        f"median={statistics.median(eigs):.3f} min={min(eigs):.3f}"
    )
    print(f"Wrote ranked stimuli to {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
