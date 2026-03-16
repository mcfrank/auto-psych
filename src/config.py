"""Configuration and paths for the auto-psych pipeline."""

from pathlib import Path
from typing import Any, Dict, Optional

# Repo root (parent of src/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level pipeline defaults (overridable via run_pipeline.py CLI)
DEFAULT_SIMULATED_N_PARTICIPANTS = 5
DEFAULT_MAX_VALIDATION_RETRIES = 3
PROJECTS_DIR = REPO_ROOT / "projects"
PROMPTS_DIR = REPO_ROOT / "prompts"
SECRETS_PATH = REPO_ROOT / ".secrets"


def project_dir(project_id: str) -> Path:
    """Return path to project directory."""
    return PROJECTS_DIR / project_id


def problem_definition_path(project_id: str) -> Path:
    """Return path to problem definition markdown file."""
    return project_dir(project_id) / "problem_definition.md"


def references_dir(project_id: str) -> Path:
    """Return path to project references directory (PDFs, .md, .txt)."""
    return project_dir(project_id) / "references"


def run_dir(project_id: str, run_id: int) -> Path:
    """Return path to run directory (e.g. projects/subjective_randomness/run1)."""
    return project_dir(project_id) / f"run{run_id}"


def run_dir_for_state(project_id: str, run_id: int, state: Optional[Dict[str, Any]] = None) -> Path:
    """Return run directory; when state has 'batch_dir', runs live under that batch."""
    if state and state.get("batch_dir"):
        return Path(state["batch_dir"]) / f"run{run_id}"
    return run_dir(project_id, run_id)


def agent_dir(project_id: str, run_id: int, agent_key: str) -> Path:
    """Return path to agent output directory (e.g. run1/1_theory)."""
    return run_dir(project_id, run_id) / agent_key


def agent_dir_for_state(project_id: str, run_id: int, agent_key: str, state: Optional[Dict[str, Any]] = None) -> Path:
    """Return agent directory; when state has 'batch_dir', runs live under that batch."""
    return run_dir_for_state(project_id, run_id, state) / agent_key


def batches_dir(project_id: str) -> Path:
    """Return path to project batches directory (e.g. projects/subjective_randomness/batches)."""
    return project_dir(project_id) / "batches"


def prompts_used_dir(project_id: str, run_id: int) -> Path:
    """Return path to prompts_used archive for a run."""
    return run_dir(project_id, run_id) / "prompts_used"


def project_prompts_dir(project_id: str) -> Path:
    """Return path to project-specific prompt overrides."""
    return project_dir(project_id) / "prompts"
