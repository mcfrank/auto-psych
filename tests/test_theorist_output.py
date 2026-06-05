"""Test theorist output validation: manifest exists, models in library, each model runs and returns a score.

DEFERRED: `src.validation.validate_theorist_output` is the LangGraph pipeline's
validator and still uses the old callable-model contract (`fn(stim, opts) ->
dict`). Under the PyMC migration first cut the LangGraph pipeline is deferred;
the active outer loop uses a PyMC-aware theory validator instead.
"""

import pytest
from pathlib import Path

pytestmark = pytest.mark.skip(
    reason="LangGraph validator deferred from PyMC migration (first cut)"
)

from src.validation import validate_theorist_output, Validated

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def test_validate_theorist_output_with_fixture_manifest(tmp_path):
    # Copy fixture manifest into a run dir and validate
    run_dir = tmp_path
    theory_dir = run_dir / "1_theory"
    theory_dir.mkdir()
    (theory_dir / "models_manifest.yaml").write_text(
        (FIXTURES_DIR / "models_manifest.yaml").read_text()
    )
    model_code = (
        "def {name}(stimulus, response_options):\n"
        "    return {{response_options[0]: 0.5, response_options[1]: 0.5}}\n"
    )
    for name in ["bayesian_fair_coin", "representativeness"]:
        (theory_dir / f"{name}.py").write_text(model_code.format(name=name))
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
        (Path(d) / "1_theory" / "models_manifest.yaml").write_text(
            "models:\n  - name: not_a_real_model\n"
        )
        result = validate_theorist_output(Path(d))
        assert not result.ok
        assert "1_theory" in result.message and (
            "not in" in result.message
            or "has no" in result.message
            or "must provide" in result.message
        )
