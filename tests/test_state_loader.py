"""Test state loader: load_state_from_run and minimal_state_for_agent produce expected paths."""

from pathlib import Path

from src.experiments.state_loader import load_state_from_run, minimal_state_for_agent
from tests.paths import FIXTURES_DIR


def test_load_state_from_run_with_fixture_run_dir(tmp_path, monkeypatch):
    """With a run dir that has fixture-like structure, load_state_from_run returns expected keys.

    The projects root is redirected at ``src.runtime.config.PROJECTS_DIR`` (the
    single place ``project_dir`` resolves it, and already the override point for
    Cloud Run's ``PIPELINE_PROJECTS_DIR``) so the run lives under ``tmp_path``
    instead of being written into — and then deleted from — the real repo tree.
    """
    monkeypatch.setattr("src.runtime.config.PROJECTS_DIR", tmp_path / "projects")

    project_id = "test_project"
    run_id = 99
    run_dir = tmp_path / "projects" / project_id / f"run{run_id}"
    (run_dir / "1_theory").mkdir(parents=True)
    (run_dir / "1_theory" / "models_manifest.yaml").write_text("models: []")
    (run_dir / "2_design").mkdir()
    (run_dir / "2_design" / "stimuli.json").write_text("[]")

    state = load_state_from_run(project_id, run_id)

    assert state["project_id"] == project_id
    assert state["run_id"] == run_id
    assert state["theorist_manifest_path"] == str(
        run_dir / "1_theory" / "models_manifest.yaml"
    )
    assert state["stimuli_path"] == str(run_dir / "2_design" / "stimuli.json")


def test_minimal_state_for_agent_theorist():
    state = minimal_state_for_agent(
        "1_theory", "subjective_randomness", 1, fixtures_dir=FIXTURES_DIR
    )
    assert state["project_id"] == "subjective_randomness"
    assert state["run_id"] == 1
    assert "problem_definition_path" in state


def test_minimal_state_for_agent_designer():
    state = minimal_state_for_agent(
        "2_design", "subjective_randomness", 1, fixtures_dir=FIXTURES_DIR
    )
    assert state["theorist_manifest_path"]
    assert Path(state["theorist_manifest_path"]).exists()
