#!/usr/bin/env python3
"""CLI for the PyMC inner model loop.

Fits a set of PyMC cognitive models to observed responses, compares them by
ELPD-LOO, optionally spawns a coding agent to propose new candidate models over
several rounds, and exports the best model.

The responses CSV must already carry the numeric feature columns the models read
through `pm.Data` (e.g. produced by a project's `preprocess.py`). When driven by
the outer loop, featurization and seeding happen in
`outer_loop.orchestrator.run_inner_model_loop_programmatic`; this CLI is for
running the inner loop directly on an already-featurized CSV + seed model set.

Usage:
  # Compute only (no agent): fit + compare the seed models, export the best.
  python3 -m src.pipelines.inner_loop.run \\
      --responses   tests/fixtures/pymc_models/responses.csv \\
      --seed-models tests/fixtures/pymc_models \\
      --results     /tmp/inner_demo \\
      --max-iterations 0

  # Full loop: also spawn the coding agent to propose candidates each round.
  python3 -m src.pipelines.inner_loop.run \\
      --responses   EXP/data/responses.csv \\
      --seed-models EXP/cognitive_models \\
      --results     EXP/model_loop \\
      --max-iterations 2 --candidate-count 3 --coding-agent claude
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro

# Ensure repo root on path so "import src..." works when run as a module/script.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from src.runtime.coding_agent import select_backend


@dataclass
class Args:
    """PyMC inner model loop: fit, compare (ELPD-LOO), and export the best model."""

    responses: Path
    """Path to a featurized responses CSV (columns match the models' pm.Data names)."""
    seed_models: Path
    """Directory of seed PyMC models (<name>.py + models_manifest.yaml)."""
    results: Path
    """Output directory for the loop (models/, model_posterior.json, best_model.py, report.md)."""
    max_iterations: int = 0
    """Candidate-generation rounds. 0 = fit/compare the seed set only (no agent spawned)."""
    candidate_count: int = 3
    """Candidate models proposed per round (only used when --max-iterations > 0)."""
    complexity_prior: float = 0.0
    """Log-prior per model = CONST * complexity (negative penalises complex models)."""
    draws: int = 500
    """MCMC posterior draws per chain."""
    tune: int = 500
    """MCMC tuning (warmup) steps per chain."""
    chains: int = 2
    """MCMC chains."""
    cache_dir: Optional[Path] = None
    """Optional directory to persist .nc fits across runs."""
    coding_agent: Optional[Literal["claude", "opencode"]] = None
    """Coding-agent backend for candidate generation. Defaults to CODING_AGENT env, then 'claude'."""
    agent_timeout_sec: int = 900
    """Per-candidate coding-agent timeout in seconds."""


def main(args: Args) -> None:
    responses_path = args.responses
    seed_models_dir = args.seed_models
    results_dir = args.results
    if not responses_path.exists():
        print(f"Error: responses CSV not found: {responses_path}", file=sys.stderr)
        sys.exit(1)
    if not (seed_models_dir / "models_manifest.yaml").exists():
        print(
            f"Error: seed-models dir has no models_manifest.yaml: {seed_models_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    backend = select_backend(args.coding_agent) if args.max_iterations > 0 else None

    result = run_pymc_inner_loop(
        responses_path,
        results_dir,
        seed_models_dir=seed_models_dir,
        max_iterations=args.max_iterations,
        candidate_count=args.candidate_count,
        complexity_prior_const=args.complexity_prior,
        cache_dir=args.cache_dir,
        agent_timeout_sec=args.agent_timeout_sec,
        backend=backend,
        fit_kwargs={"draws": args.draws, "tune": args.tune, "chains": args.chains},
    )

    best = result["best_model"]
    print(
        f"\nBest model: {best} "
        f"(posterior={result['posteriors'][best]:.3f}, elpd_loo={result['elpd_loo'][best]:.2f})",
        flush=True,
    )
    print(f"Artifacts: {results_dir}", flush=True)
    print(json.dumps(result["posteriors"], indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
