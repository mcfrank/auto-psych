"""Campaign configuration: the pinned settings of one improvement campaign.

``start_campaign.sh`` writes ``<campaign root>/campaign.env`` once; every
review job reads it back through :meth:`Campaign.load`, so an iteration never
depends on the environment of the shell that started the campaign. The file is
plain ``KEY=value`` (bash-sourceable) so it is readable by both the sbatch
wrapper and this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# campaign.env keys every campaign must pin (fail loudly if one is missing).
REQUIRED_KEYS = (
    "CAMPAIGN_NAME",
    "SOURCE_REPO",
    "BASE_BRANCH",
    "BASELINE_ROOTS",
    "MAX_ITERATIONS",
    "REVIEW_MODEL",
    "REVIEW_MAX_TURNS",
    "REVIEW_MAX_BUDGET_USD",
    "REVIEW_TIMEOUT_SEC",
    "MAX_REVIEW_REPAIRS",
)

# campaign.env keys that pin the sweep the agent's next_run.env starts from,
# mapped to the env-var names submit_holdout_test_retest.sh reads.
SWEEP_DEFAULT_KEYS = {
    "SWEEP_N_REPEATS": "N_REPEATS",
    "SWEEP_BASE_SEED": "BASE_SEED",
    "SWEEP_MAX_PARALLEL": "MAX_PARALLEL",
    "SWEEP_GT_MODELS": "GT_MODELS",
    "SWEEP_CONFIG": "CONFIG",
    "SWEEP_SEED_MODELS_REL": "SEED_MODELS_REL",
}

# Slurm settings of the review job itself (optional, with defaults).
REVIEW_SLURM_DEFAULTS = {
    "REVIEW_PARTITION": "normal",
    "REVIEW_TIME": "08:00:00",
    "REVIEW_CPUS": "4",
    "REVIEW_MEM": "16GB",
}


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a ``KEY=value`` file (optional ``export`` prefix and quotes)."""
    values: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{lineno}: expected KEY=value, got {raw!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key.isidentifier():
            raise ValueError(f"{path}:{lineno}: invalid key {key!r}")
        values[key] = value
    return values


@dataclass(frozen=True)
class Campaign:
    """The pinned settings of one campaign, read from ``campaign.env``."""

    root: Path
    name: str
    source_repo: Path
    """The checkout iteration 1 branches from (and whose driver scripts run)."""
    base_branch: str
    baseline_roots: tuple[Path, ...]
    """Finished sweep roots the first iteration reviews; the first one is the
    reference every later sweep is compared against."""
    max_iterations: int
    review_model: str
    review_max_turns: int
    review_max_budget_usd: float
    review_timeout_sec: int
    max_review_repairs: int
    sweep_defaults: dict[str, str]
    """Env for submit_holdout_test_retest.sh that next_run.env overrides."""
    review_slurm: dict[str, str]
    """Partition/time/cpus/mem of the review job (REVIEW_* keys)."""
    driver_sbatch: Path
    """The review_iteration.sbatch that chained review jobs run."""

    @classmethod
    def load(cls, root: Path | str) -> "Campaign":
        root = Path(root)
        env_path = root / "campaign.env"
        if not env_path.is_file():
            raise FileNotFoundError(f"No campaign.env under {root}")
        values = parse_env_file(env_path)
        missing = [k for k in REQUIRED_KEYS if not values.get(k)]
        if missing:
            raise ValueError(f"{env_path} is missing required keys: {missing}")
        source_repo = Path(values["SOURCE_REPO"])
        baseline_roots = tuple(Path(p) for p in values["BASELINE_ROOTS"].split())
        if not baseline_roots:
            raise ValueError(f"{env_path}: BASELINE_ROOTS names no sweep roots")
        sweep_defaults = {
            env_name: values[key]
            for key, env_name in SWEEP_DEFAULT_KEYS.items()
            if values.get(key)
        }
        review_slurm = {
            key: values.get(key) or default
            for key, default in REVIEW_SLURM_DEFAULTS.items()
        }
        driver_sbatch = Path(
            values.get("DRIVER_SBATCH")
            or source_repo / "scripts" / "recovery_improvement" / "review_iteration.sbatch"
        )
        return cls(
            root=root,
            name=values["CAMPAIGN_NAME"],
            source_repo=source_repo,
            base_branch=values["BASE_BRANCH"],
            baseline_roots=baseline_roots,
            max_iterations=int(values["MAX_ITERATIONS"]),
            review_model=values["REVIEW_MODEL"],
            review_max_turns=int(values["REVIEW_MAX_TURNS"]),
            review_max_budget_usd=float(values["REVIEW_MAX_BUDGET_USD"]),
            review_timeout_sec=int(values["REVIEW_TIMEOUT_SEC"]),
            max_review_repairs=int(values["MAX_REVIEW_REPAIRS"]),
            sweep_defaults=sweep_defaults,
            review_slurm=review_slurm,
            driver_sbatch=driver_sbatch,
        )

    def iteration_dir(self, iteration: int) -> Path:
        return self.root / f"iter{iteration}"

    def branch_name(self, iteration: int) -> str:
        return f"recovery-improvement/{self.name}/iter{iteration}"

    @property
    def journal_path(self) -> Path:
        return self.root / "journal.md"

    @property
    def stop_path(self) -> Path:
        """Touched by stop_campaign.sh; a review job that finds it exits at once."""
        return self.root / "STOP"
