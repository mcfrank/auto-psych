"""Fast tests for the recovery text builders in `reporting.py`."""

from __future__ import annotations

import pytest

from src.subjective_randomness.reporting import (
    model_recovery_text,
    parameter_recovery_text,
    recovery_note,
)

SAMPLED_REPORT = {
    "model": "demo",
    "n_repeats": 2,
    "param_ranges": {"beta": [0.2, 12.0]},
    "runs": [
        {
            "repeat": 0,
            "true_params": {"beta": 1.0},
            "posterior": {"beta": {"mean": 1.2, "q025": 0.8, "q975": 1.6}},
        },
        {
            "repeat": 1,
            "true_params": {"beta": 8.0},
            "posterior": {"beta": {"mean": 7.5, "q025": 6.9, "q975": 8.1}},
        },
    ],
}

CONFUSION = {
    "seed_models": ["A", "B"],
    "generator": "pymc",
    "generating": [
        {
            "generating_model": "A",
            "posteriors": {"A": 0.8, "B": 0.2},
            "elpd_loo": {"A": -10.0, "B": -12.0},
        },
        {
            "generating_model": "B",
            "posteriors": {"A": 0.6, "B": 0.4},
            "elpd_loo": {"A": -9.0, "B": -9.5},
        },
    ],
}


def test_parameter_recovery_text_includes_model_repeats_and_pearson():
    text = parameter_recovery_text(SAMPLED_REPORT)
    assert "Parameter recovery — model: demo" in text
    assert "repeats: 2" in text
    assert "pearson_r" in text
    assert "beta" in text


def test_model_recovery_text_includes_accuracy_and_per_model_rows():
    text = model_recovery_text(CONFUSION)
    assert "Closed-ended model recovery — generator: pymc" in text
    assert "posterior accuracy: 0.50 (1/2)" in text
    assert "<- mis-recovered" in text  # B's data was won by A


@pytest.mark.parametrize(
    "correct, distinguishable, expected",
    [
        (False, False, "   <- mis-recovered (but tied: not distinguishable)"),
        (False, True, "   <- mis-recovered"),
        (False, None, "   <- mis-recovered"),  # no comparison -> plain mis-recovered
        (True, False, "   <- recovered, but tied with runner-up"),
        (True, True, ""),
        (True, None, ""),  # no comparison -> no annotation
    ],
)
def test_recovery_note_branches(correct, distinguishable, expected):
    note = recovery_note(
        {"correct_posterior": correct, "winner_distinguishable": distinguishable}
    )
    assert note == expected
