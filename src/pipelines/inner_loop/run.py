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
from typing import List, Literal, Optional

import tyro
import yaml

# Ensure repo root on path so "import src..." works when run as a module/script.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.models.mcmc_defaults import (
    PRODUCTION_CHAINS,
    PRODUCTION_DRAWS,
    PRODUCTION_TUNE,
)
from src.pipelines.inner_loop.pymc_orchestrator import (
    DEFAULT_COMPLEXITY_PRIOR_CONST,
    DEFAULT_NOVELTY_RMSE_THRESHOLD,
    DEFAULT_PRUNE_DSE_MULTIPLIER,
    DEFAULT_PRUNE_WEIGHT_FLOOR,
    run_pymc_inner_loop,
)
from src.runtime.coding_agent import select_backend


def load_hints_file(path: Path) -> List[str]:
    """Load exploration hints: a YAML list of strings, one hint per entry."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hints file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(
        isinstance(h, str) and h.strip() for h in data
    ):
        raise ValueError(
            f"Hints file must be a YAML list of non-empty strings: {path}"
        )
    return [h.strip() for h in data]


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
    complexity_prior: float = DEFAULT_COMPLEXITY_PRIOR_CONST
    """Log-prior per model = CONST * non-comment line count (negative penalises
    complex models). Defaults to a gentle Occam backstop; pass 0.0 to disable."""
    draws: int = PRODUCTION_DRAWS
    """MCMC posterior draws per chain (src.models.mcmc_defaults; pass lower values for quick local runs)."""
    tune: int = PRODUCTION_TUNE
    """MCMC tuning (warmup) steps per chain."""
    chains: int = PRODUCTION_CHAINS
    """MCMC chains."""
    cache_dir: Optional[Path] = None
    """Optional directory to persist .nc fits across runs."""
    coding_agent: Optional[Literal["claude", "opencode"]] = None
    """Coding-agent backend for candidate generation. Defaults to CODING_AGENT env, then 'opencode'."""
    agent_timeout_sec: int = 900
    """Per-candidate coding-agent timeout in seconds."""
    hints_file: Optional[Path] = None
    """YAML list of exploration hints cycled across a round's candidates
    (default: the built-in DEFAULT_CANDIDATE_HINTS lens battery)."""
    novelty_rmse_threshold: float = DEFAULT_NOVELTY_RMSE_THRESHOLD
    """Reject a candidate whose p_left is within this RMSE of an admitted
    model's on the observed stimuli (0 disables the novelty gate)."""
    prune_dse_multiplier: float = DEFAULT_PRUNE_DSE_MULTIPLIER
    """Prune agent models with elpd_diff > multiplier*dse AND stacking weight
    below the floor after each scoring pass (0 disables pruning)."""
    prune_weight_floor: float = DEFAULT_PRUNE_WEIGHT_FLOOR
    """Stacking-weight floor for pruning (see --prune-dse-multiplier)."""
    candidate_parallelism: Optional[int] = None
    """Concurrent candidate agents per round (default: all of the round's
    candidates at once; 1 = sequential)."""


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
    hints = load_hints_file(args.hints_file) if args.hints_file else None

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
        candidate_hints=hints,
        novelty_rmse_threshold=args.novelty_rmse_threshold,
        prune_dse_multiplier=args.prune_dse_multiplier,
        prune_weight_floor=args.prune_weight_floor,
        candidate_parallelism=args.candidate_parallelism,
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
