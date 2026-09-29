"""Configuration and paths for the auto-psych pipeline."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pyprojroot import here

# The one repo root: everything under src/ imports it from here rather than
# counting `Path(__file__).parents[N]` for itself (nine such counts had drifted
# into three different spellings, each silently wrong if its file ever moved).
# Modules that must bootstrap `sys.path` before `src` is importable call
# pyprojroot's here() directly — the same resolution as this one.
REPO_ROOT = here()

# Top-level defaults retained for shared support modules.
DEFAULT_SIMULATED_N_PARTICIPANTS = 5
DEFAULT_MAX_VALIDATION_RETRIES = 3
# An agent call that ends on its account's usage or rate limit
# (src/runtime/usage_limits.py) waits for the limit to lift and runs again,
# for at most AGENT_USAGE_LIMIT_MAX_WAIT_SEC in all; then the run raises
# AgentUsageLimitExceeded, and the sweep's retry job resumes the cell later.
AGENT_USAGE_LIMIT_MAX_WAIT_SEC = 12 * 3600
# The wait runs until this long after the stated reset time...
AGENT_USAGE_LIMIT_RESET_MARGIN_SEC = 2 * 60
# ...or, when the message states none (an API rate limit, say), this long.
AGENT_USAGE_LIMIT_FALLBACK_WAIT_SEC = 10 * 60
# Run *outputs* of the agent pipeline (projects/<id>/run<N>/…).
# Override via PIPELINE_PROJECTS_DIR for Cloud Run (e.g. /app/projects)
PROJECTS_DIR = Path(os.environ.get("PIPELINE_PROJECTS_DIR", REPO_ROOT / "projects"))
# Project *assets* — problem_definition.md, ground_truth_models.py,
# seed_models/, references/, prolific_config.yaml. These ship with the package
# next to the outer loop that reads them, and are deliberately NOT under
# PROJECTS_DIR: the two roots each held a copy of the assets for a while, and the
# root copies silently went stale while the pipeline ran the ones under src/.
PROJECT_ASSETS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects"
PROMPTS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "prompts"
SECRETS_PATH = REPO_ROOT / ".secrets"


def project_dir(project_id: str) -> Path:
    """Return path to project run-output directory."""
    return PROJECTS_DIR / project_id


def project_assets_dir(project_id: str) -> Path:
    """Return path to the project's checked-in assets directory."""
    return PROJECT_ASSETS_DIR / project_id


def problem_definition_path(project_id: str) -> Path:
    """Return path to problem definition markdown file."""
    return project_assets_dir(project_id) / "problem_definition.md"


def references_dir(project_id: str) -> Path:
    """Return path to project references directory (PDFs, .md, .txt)."""
    return project_assets_dir(project_id) / "references"


def run_dir(project_id: str, run_id: int) -> Path:
    """Return path to run directory (e.g. projects/subjective_randomness/run1)."""
    return project_dir(project_id) / f"run{run_id}"


def run_dir_for_state(
    project_id: str, run_id: int, state: Optional[Dict[str, Any]] = None
) -> Path:
    """Return run directory; when state has 'batch_dir', runs live under that batch."""
    if state and state.get("batch_dir"):
        return Path(state["batch_dir"]) / f"run{run_id}"
    return run_dir(project_id, run_id)


def agent_dir(project_id: str, run_id: int, agent_key: str) -> Path:
    """Return path to agent output directory (e.g. run1/1_theory)."""
    return run_dir(project_id, run_id) / agent_key


def prompts_used_dir(project_id: str, run_id: int) -> Path:
    """Return path to prompts_used archive for a run."""
    return run_dir(project_id, run_id) / "prompts_used"


def project_prompts_dir(project_id: str) -> Path:
    """Return path to project-specific prompt overrides."""
    return project_assets_dir(project_id) / "prompts"
