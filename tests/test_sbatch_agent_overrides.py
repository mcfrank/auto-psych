"""The holdout array sbatch must allow the agent backend and model to be
overridden via AGENT_BACKEND and AGENT_MODEL environment variables, so a
Fable 5.1 (or any other backend/model) smoke can reuse the same scripts."""

from pathlib import Path

from pyprojroot import here


SBATCH_PATH = here() / "scripts/subjective_randomness/slurm/holdout_recovery_array.sbatch"
SUBMIT_PATH = here() / "scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh"


def test_sbatch_uses_agent_backend_env_var():
    """The --backend flag defaults to opencode but honors AGENT_BACKEND."""
    text = SBATCH_PATH.read_text(encoding="utf-8")
    assert '${AGENT_BACKEND:-opencode}' in text, (
        "holdout_recovery_array.sbatch must use ${AGENT_BACKEND:-opencode} "
        "for the --backend flag so callers can override the agent backend"
    )


def test_sbatch_uses_agent_model_env_var():
    """The --agent-model flag is passed when AGENT_MODEL is set."""
    text = SBATCH_PATH.read_text(encoding="utf-8")
    assert 'AGENT_MODEL' in text, (
        "holdout_recovery_array.sbatch must pass --agent-model from "
        "AGENT_MODEL env var when set"
    )


def test_submit_script_exports_agent_overrides():
    """The submit script passes AGENT_BACKEND and AGENT_MODEL through."""
    text = SUBMIT_PATH.read_text(encoding="utf-8")
    assert 'AGENT_BACKEND' in text, (
        "submit_holdout_test_retest.sh must export AGENT_BACKEND if set"
    )
    assert 'AGENT_MODEL' in text, (
        "submit_holdout_test_retest.sh must export AGENT_MODEL if set"
    )
