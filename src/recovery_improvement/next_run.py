"""The agent's ``next_run.env`` contract: which sweep to launch next.

The review agent declares the sweep that should test its tweaks by writing
``KEY=value`` lines to ``<iteration dir>/next_run.env``. Only the knobs listed
in :data:`ALLOWED_KEYS` are accepted — they are exactly the env vars
``scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh`` reads —
and every line is validated here before anything is submitted, so a typo costs
a repair round rather than a dead 20-task array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.recovery_improvement.campaign import Campaign, parse_env_file

ALLOWED_KEYS: dict[str, str] = {
    "CONFIG": "holdout config, relative to the repo (must exist there)",
    "GT_MODELS": "space-separated held-out ground truths (must equal the config's gt_models)",
    "N_REPEATS": "repeats per ground truth (test-retest design; the baseline used 5)",
    "BASE_SEED": "repeat r uses seed BASE_SEED+r (the baseline used 100; keep it for paired comparisons)",
    "MAX_PARALLEL": "simultaneous array tasks",
    "SEED_MODELS_REL": "GT/baseline registry dir, relative to the repo",
    "N_EXPERIMENTS": "outer-loop experiments per run",
    "N_PARTICIPANTS": "synthetic participants per experiment",
    "INNER_LOOP_ITERATIONS": "inner-loop candidate rounds (0 = seed set only)",
    "INNER_LOOP_CANDIDATES": "candidate models per inner-loop round",
    "DESIGN_N_EIG": "stimuli chosen by max joint EIG per experiment",
    "DESIGN_N_RANDOM": "uniform-coverage stimuli per experiment",
    "DRAWS": "MCMC draws per chain",
    "TUNE": "MCMC tuning steps per chain",
    "CHAINS": "MCMC chains",
    "AGENT_TIMEOUT_SEC": "per-agent timeout inside the sweep",
    "SMOKE": "set to 1 for a one-task cheap pre-flight instead of a real sweep",
    "NOTE": "free text: what this sweep tests (recorded in the journal, not passed on)",
}
INT_KEYS = {
    "N_REPEATS", "BASE_SEED", "MAX_PARALLEL", "N_EXPERIMENTS", "N_PARTICIPANTS",
    "INNER_LOOP_ITERATIONS", "INNER_LOOP_CANDIDATES", "DESIGN_N_EIG",
    "DESIGN_N_RANDOM", "DRAWS", "TUNE", "CHAINS", "AGENT_TIMEOUT_SEC",
}
# Keys that describe the sweep but are not forwarded to the launcher.
LOCAL_KEYS = {"NOTE"}


@dataclass(frozen=True)
class NextRun:
    """A validated next_run.env."""

    values: dict[str, str] = field(default_factory=dict)
    """Launcher env overrides (no NOTE)."""
    note: str = ""


def describe_allowed_keys() -> str:
    """Markdown bullet list of the accepted keys, for the agent's prompt."""
    return "\n".join(f"- `{key}` — {desc}" for key, desc in ALLOWED_KEYS.items())


def _config_gt_models(repo: Path, config_rel: str) -> list[str]:
    config = yaml.safe_load((repo / config_rel).read_text(encoding="utf-8")) or {}
    return sorted((config.get("gt_models") or {}).keys())


def parse_next_run(path: Path, *, repo: Path, defaults: dict[str, str]) -> NextRun:
    """Parse and validate ``next_run.env``; raise ValueError listing every problem.

    ``defaults`` is the campaign's sweep default env; the config/ground-truth
    consistency check runs on the merged view, so a next_run.env that changes
    only CONFIG is still checked against the default GT_MODELS (and vice versa).
    """
    raw = parse_env_file(path)
    problems: list[str] = []
    for key, value in raw.items():
        if key not in ALLOWED_KEYS:
            problems.append(f"unknown key {key!r} (allowed: {', '.join(ALLOWED_KEYS)})")
        elif key in INT_KEYS:
            if not value.lstrip("-").isdigit():
                problems.append(f"{key} must be an integer, got {value!r}")
            elif key != "BASE_SEED" and int(value) < 0:
                problems.append(f"{key} must be non-negative, got {value}")
        elif not value:
            problems.append(f"{key} is empty")
    if raw.get("N_REPEATS", "1").isdigit() and int(raw.get("N_REPEATS", "1")) < 1:
        problems.append("N_REPEATS must be at least 1")

    merged = {**defaults, **{k: v for k, v in raw.items() if k in ALLOWED_KEYS}}
    config_rel = merged.get("CONFIG")
    if config_rel:
        if not (repo / config_rel).is_file():
            problems.append(f"CONFIG {config_rel!r} does not exist in {repo}")
        elif merged.get("GT_MODELS"):
            expected = _config_gt_models(repo, config_rel)
            declared = sorted(merged["GT_MODELS"].split())
            if declared != expected:
                problems.append(
                    f"GT_MODELS {declared} != gt_models of {config_rel} {expected} "
                    "(the sweep's setup job would abort)"
                )
    if problems:
        raise ValueError(
            f"{path} is invalid:\n" + "\n".join(f"  - {p}" for p in problems)
        )
    values = {k: v for k, v in raw.items() if k not in LOCAL_KEYS}
    return NextRun(values=values, note=raw.get("NOTE", ""))


def sweep_env(next_run: NextRun, campaign: Campaign, *, repo: Path, work_root: Path) -> dict[str, str]:
    """The env submit_holdout_test_retest.sh gets: defaults, overrides, locations."""
    return {
        **campaign.sweep_defaults,
        **next_run.values,
        "REPO": str(repo),
        "WORK_ROOT": str(work_root),
    }
