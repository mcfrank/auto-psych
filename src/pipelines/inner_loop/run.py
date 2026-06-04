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

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root on path so "import src..." works when run as a module/script.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from src.runtime.coding_agent import select_backend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyMC inner model loop: fit, compare (ELPD-LOO), and export the best model."
    )
    parser.add_argument(
        "--responses", required=True,
        help="Path to a featurized responses CSV (columns match the models' pm.Data names).",
    )
    parser.add_argument(
        "--seed-models", required=True,
        help="Directory of seed PyMC models (<name>.py + models_manifest.yaml).",
    )
    parser.add_argument(
        "--results", required=True,
        help="Output directory for the loop (models/, model_posterior.json, best_model.py, report.md).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=0, metavar="N",
        help="Candidate-generation rounds. 0 = fit/compare the seed set only (no agent spawned).",
    )
    parser.add_argument(
        "--candidate-count", type=int, default=3, metavar="N",
        help="Candidate models proposed per round (only used when --max-iterations > 0).",
    )
    parser.add_argument(
        "--complexity-prior", type=float, default=0.0, metavar="CONST",
        help="Log-prior per model = CONST * complexity (negative penalises complex models).",
    )
    parser.add_argument("--draws", type=int, default=500, help="MCMC posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=500, help="MCMC tuning (warmup) steps per chain.")
    parser.add_argument("--chains", type=int, default=2, help="MCMC chains.")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Optional directory to persist .nc fits across runs.",
    )
    parser.add_argument(
        "--coding-agent", choices=["claude", "opencode"], default=None,
        help="Coding-agent backend for candidate generation. Defaults to CODING_AGENT env, then 'claude'.",
    )
    parser.add_argument(
        "--agent-timeout-sec", type=int, default=900,
        help="Per-candidate coding-agent timeout in seconds.",
    )
    args = parser.parse_args()

    responses_path = Path(args.responses)
    seed_models_dir = Path(args.seed_models)
    results_dir = Path(args.results)
    if not responses_path.exists():
        print(f"Error: responses CSV not found: {responses_path}", file=sys.stderr)
        sys.exit(1)
    if not (seed_models_dir / "models_manifest.yaml").exists():
        print(f"Error: seed-models dir has no models_manifest.yaml: {seed_models_dir}", file=sys.stderr)
        sys.exit(1)

    backend = select_backend(args.coding_agent) if args.max_iterations > 0 else None

    result = run_pymc_inner_loop(
        responses_path,
        results_dir,
        seed_models_dir=seed_models_dir,
        max_iterations=args.max_iterations,
        candidate_count=args.candidate_count,
        complexity_prior_const=args.complexity_prior,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
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
    main()
