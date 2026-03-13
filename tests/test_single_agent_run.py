"""Test running a single agent with minimal state from fixtures and that its output passes the validator."""

import pytest
from pathlib import Path

from src.state_loader import minimal_state_for_agent
from src.agents.theorist import run_theorist
from src.validation import validate_theorist_output

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture
def project_id():
    return "subjective_randomness"


def test_theorist_run_with_fixtures_then_validate(tmp_path, project_id):
    """Run theorist with minimal state (fixtures for problem def); then validate output."""
    # Create a minimal project/run layout under tmp_path so theorist can write
    projects_dir = tmp_path / "projects" / project_id
    projects_dir.mkdir(parents=True)
    (projects_dir / "problem_definition.md").write_text((FIXTURES_DIR / "problem_definition.md").read_text())
    run_id = 1
    run_dir = projects_dir / "run1"
    run_dir.mkdir()
    (run_dir / "1theorist").mkdir()
    (run_dir / "prompts_used").mkdir()
    (run_dir / "prompts_used" / "1theorist.md").write_text("You are the theorist. Output YAML with models and rationale.")

    # We need to patch config to use tmp_path as REPO_ROOT so paths resolve
    import src.config as config
    orig_projects = config.PROJECTS_DIR
    config.PROJECTS_DIR = tmp_path / "projects"
    try:
        state = minimal_state_for_agent("1theorist", project_id, run_id, fixtures_dir=FIXTURES_DIR)
        state["problem_definition_path"] = str(projects_dir / "problem_definition.md")
        # Run theorist (may call LLM; if no key, fallback will write default manifest)
        result = run_theorist(state)
    finally:
        config.PROJECTS_DIR = orig_projects

    assert "theorist_manifest_path" in result
    manifest_path = Path(result["theorist_manifest_path"])
    assert manifest_path.exists()

    # Validate: run_dir for validator is the run directory
    validation = validate_theorist_output(run_dir)
    assert validation.ok, validation.message
