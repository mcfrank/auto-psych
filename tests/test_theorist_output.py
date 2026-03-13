"""Test theorist output validation: manifest exists, models in library, each model runs and returns a score."""

import pytest
from pathlib import Path

from src.validation import validate_theorist_output, Validated

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def test_validate_theorist_output_with_fixture_manifest(tmp_path):
    # Copy fixture manifest into a run dir and validate
    run_dir = tmp_path
    (run_dir / "1_theory").mkdir()
    (run_dir / "1_theory" / "models_manifest.yaml").write_text((FIXTURES_DIR / "models_manifest.yaml").read_text())
    result = validate_theorist_output(run_dir)
    assert result.ok, result.message
    assert "model_names" in (result.details or {})


def test_validate_theorist_output_missing_manifest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        result = validate_theorist_output(Path(d))
        assert not result.ok
        assert "not found" in result.message.lower()


def test_validate_theorist_output_invalid_model_name():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "1_theory").mkdir()
        (Path(d) / "1_theory" / "models_manifest.yaml").write_text("models:\n  - name: not_a_real_model\n")
        result = validate_theorist_output(Path(d))
        assert not result.ok
        assert "not in MODEL_LIBRARY" in result.message
