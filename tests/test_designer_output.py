"""Test experiment designer output validation and EIG helper."""

import pytest
from pathlib import Path

from src.agents.experiment_designer import expected_information_gain
from src.validation import validate_designer_output

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def test_validate_designer_output_with_fixture(tmp_path):
    (tmp_path / "2_design").mkdir()
    (tmp_path / "2_design" / "stimuli.json").write_text((FIXTURES_DIR / "stimuli.json").read_text())
    result = validate_designer_output(tmp_path)
    assert result.ok, result.message
    assert result.details and result.details.get("n_stimuli") == 2


def test_validate_designer_output_missing_file():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        result = validate_designer_output(Path(d))
        assert not result.ok
        assert "not found" in result.message.lower()


def test_validate_designer_output_invalid_stimulus_missing_keys():
    import tempfile
    import json
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "2_design").mkdir()
        (Path(d) / "2_design" / "stimuli.json").write_text(json.dumps([{"wrong": "key"}]))
        result = validate_designer_output(Path(d))
        assert not result.ok
        assert "sequence_a" in result.message or "sequence_b" in result.message


def test_expected_information_gain_helper():
    """EIG helper is non-negative; discriminating stimulus gives positive EIG."""
    model_names = ["bayesian_fair_coin", "representativeness", "alternation"]
    # Stimulus where models can disagree: very alternating vs very blocky
    stimulus = ("HTHTHTHT", "HHHHHHHH")
    eig = expected_information_gain(stimulus, model_names, theorist_dir=None)
    assert eig >= 0.0
    assert eig <= 2.0  # at most log2(n_models) bits
    # Should be positive when models differ in predictions
    assert eig > 0.01, "EIG should be positive for discriminating stimulus"


def test_validate_designer_output_missing_eig():
    import tempfile
    import json
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "2_design").mkdir()
        (Path(d) / "2_design" / "stimuli.json").write_text(
            json.dumps([{"sequence_a": "HHTHTTHT", "sequence_b": "HTHTHTHT"}])
        )
        result = validate_designer_output(Path(d))
        assert not result.ok
        assert "eig" in result.message.lower()
