"""CLI for the RSA inner loop on a fixed dataset.

    # Score the seed models only (no agents):
    uv run python -m src.rsa.loop.run --results data/rsa/loop_seeds --max-iterations 0

    # One round of three agents (Gemini through opencode by default):
    uv run python -m src.rsa.loop.run --results data/rsa/loop_smoke \
        --max-iterations 1 --candidate-count 3 --no-sandbox

Sandboxing (bubblewrap) is on by default, as for every loop agent; turn it
off only in a disposable container (e.g. a cloud session) that holds nothing
an agent should not see.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import tyro
import yaml

from src.models.model_manifest import read_manifest_entries
from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC
from src.rsa.loop.novelty import DEFAULT_NOVELTY_RMSE_THRESHOLD
from src.rsa.loop.orchestrator import (
    DEFAULT_PRUNE_DSE_MULTIPLIER,
    LoopConfig,
    RSALoop,
    coding_agent_spawner,
)
from src.runtime.config import PROJECT_ASSETS_DIR
from src.runtime.token_usage import start_usage_log, write_usage_report


DEFAULT_AGENT_MODEL = "google/gemini-3.8-flash"


@dataclass
class Args:
    results: Path
    """Output directory (model_loop layout; see src/rsa/loop/orchestrator.py)."""
    responses: Path = DEFAULT_TRIALS_CSV
    """Trials CSV in the canonical pragmods schema (included forced-choice rows are used)."""
    seed_models: Path = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
    exclude_seeds: List[str] = field(default_factory=list)
    """Seed models left out of the starting set (a recovery run's ground truth
    and its near-twins). The run's seed pool is written to <results>/seed_pool."""
    max_iterations: int = 1
    candidate_count: int = 3
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0
    """The run's random seed: seeds every NUTS fit (replicate runs differ by it,
    besides the agents' own sampling)."""
    novelty_rmse_threshold: float = DEFAULT_NOVELTY_RMSE_THRESHOLD
    prune_dse_multiplier: float = DEFAULT_PRUNE_DSE_MULTIPLIER
    fit_time_limit_sec: float = FIT_TIME_LIMIT_SEC
    coding_agent: Optional[Literal["claude", "opencode"]] = None
    """Agent backend; defaults to CODING_AGENT, then opencode (Gemini)."""
    agent_model: Optional[str] = DEFAULT_AGENT_MODEL
    """opencode provider/model for the candidate agents (PI decision 2026-10-07:
    Gemini 3.8 Flash, not the launcher's older Pro default)."""
    agent_timeout_sec: int = 2400
    """Per-agent limit. Smoke test 1 measured a self-check at 2-10 min (since the
    2026-10-07 fit optimisation ~90 s on the combined data) and a careful agent runs 1-3;
    at 1,200 s the smoke test's agents ran out mid-check."""
    agent_root: Optional[Path] = None
    """Directory agents run from (the scrubbed agent tree on the cluster; default: the repo)."""
    no_sandbox: bool = False
    """Run agents without bubblewrap. Only in a disposable container."""
    title: str = "RSA inner loop"


def seed_pool(seed_models: Path, exclude: List[str], results: Path) -> Path:
    """The starting seed set: ``seed_models`` minus ``exclude`` (written under results)."""
    if not exclude:
        return Path(seed_models)
    entries = read_manifest_entries(seed_models)
    names = {e["name"] for e in entries}
    unknown = sorted(set(exclude) - names)
    if unknown:
        raise ValueError(f"--exclude-seeds names no seed model: {unknown}")
    kept = [e for e in entries if e["name"] not in set(exclude)]
    if not kept:
        raise ValueError("--exclude-seeds leaves no seed model")
    pool = Path(results) / "seed_pool"
    pool.mkdir(parents=True, exist_ok=True)
    for e in kept:
        shutil.copyfile(Path(seed_models) / f"{e['name']}.py", pool / f"{e['name']}.py")
    (pool / "models_manifest.yaml").write_text(yaml.safe_dump({"models": kept}, sort_keys=False))
    return pool


def main(args: Args) -> int:
    results = Path(args.results)
    cfg = LoopConfig(
        responses_path=args.responses, seed_models_dir=seed_pool(args.seed_models, args.exclude_seeds, results), results_dir=results,
        max_iterations=args.max_iterations, candidate_count=args.candidate_count,
        settings=FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                             seed=args.seed),
        novelty_threshold=args.novelty_rmse_threshold, prune_dse_multiplier=args.prune_dse_multiplier,
        fit_time_limit_sec=args.fit_time_limit_sec, report_title=args.title,
    )
    spawn = coding_agent_spawner(
        models_dir=results / "models", responses_path=results / "responses.csv",
        timeout_sec=args.agent_timeout_sec, backend=args.coding_agent, model=args.agent_model,
        agent_root=args.agent_root, sandbox=not args.no_sandbox,
    )
    # The agents' spend, written in a finally so an aborted loop still accounts for it.
    usage_marker = start_usage_log(results / "token_usage.jsonl")
    try:
        final = RSALoop(cfg, spawn).run()
    finally:
        write_usage_report(results, usage_marker, heading="RSA inner loop")
    print(f"best model: {final['best_model']}; live: {sorted(final['standing'])}")
    print(f"report: {results / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
