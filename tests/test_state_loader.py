"""Test state loader: load_state_from_run and minimal_state_for_agent produce expected paths."""

import pytest
from pathlib import Path

from src.state_loader import load_state_from_run, minimal_state_for_agent
from src.config import REPO_ROOT

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def test_load_state_from_run_with_fixture_run_dir():
    """With a run dir that has fixture-like structure, load_state_from_run returns expected keys."""
    import tempfile
    import shutil
    run_dir = Path(tempfile.mkdtemp())
    try:
        (run_dir / "1theorist").mkdir()
        (run_dir / "1theorist" / "models_manifest.yaml").write_text("models: []")
        (run_dir / "2experiment_designer").mkdir()
        (run_dir / "2experiment_designer" / "stimuli.json").write_text("[]")
        # We need a project dir that contains this run
        project_id = "test_project"
        run_id = 99
        projects_dir = REPO_ROOT / "projects"
        project_dir = projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        run_actual = project_dir / f"run{run_id}"
        if run_actual.exists():
            shutil.rmtree(run_actual)
        shutil.copytree(run_dir, run_actual)
        state = load_state_from_run(project_id, run_id)
        assert state["project_id"] == project_id
        assert state["run_id"] == run_id
        assert "theorist_manifest_path" in state
        assert "stimuli_path" in state
        shutil.rmtree(run_actual, ignore_errors=True)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_minimal_state_for_agent_theorist():
    state = minimal_state_for_agent("1theorist", "subjective_randomness", 1, fixtures_dir=FIXTURES_DIR)
    assert state["project_id"] == "subjective_randomness"
    assert state["run_id"] == 1
    assert "problem_definition_path" in state


def test_minimal_state_for_agent_designer():
    state = minimal_state_for_agent("2experiment_designer", "subjective_randomness", 1, fixtures_dir=FIXTURES_DIR)
    assert state["theorist_manifest_path"]
    assert Path(state["theorist_manifest_path"]).exists()
