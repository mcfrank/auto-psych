"""Fast tests for the shared recovery text/figure builders in `reporting.py`.

The heavier output paths (figures, CSVs) are exercised through the
`analyze_recovery.py` CLI tests; these pin the text-level behaviors directly.
"""

from __future__ import annotations

import pytest

from src.subjective_randomness.reporting import (
    model_recovery_text,
    parameter_recovery_text,
    plot_selection_comparison_models,
    plot_selection_comparison_parameters,
    recovery_note,
    selection_comparison_model_text,
    selection_comparison_parameter_text,
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


PARAM_COMPARISON = {
    "model": "demo",
    "n_repeats": 2,
    "n_stimuli": 2,
    "n_participants": 10,
    "points_per_dim": 3,
    "seed": 0,
    "pool_size": 20,
    "param_ranges": {"beta": [0.2, 12.0]},
    "arms": {
        "eig": {
            "stimuli": [
                {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT", "eig": 0.35},
                {"sequence_a": "HHHHHH", "sequence_b": "HTTHTH", "eig": 0.27},
            ],
            "mean_stimulus_eig": 0.31,
            "runs": [
                {
                    "repeat": 0,
                    "true_params": {"beta": 1.0},
                    "posterior_mean": {"beta": 1.2},
                    "posterior_sd": {"beta": 0.5},
                    "final_entropy_bits": 2.0,
                },
                {
                    "repeat": 1,
                    "true_params": {"beta": 8.0},
                    "posterior_mean": {"beta": 7.5},
                    "posterior_sd": {"beta": 0.6},
                    "final_entropy_bits": 2.2,
                },
            ],
            "summary": {
                "beta": {
                    "pearson_r": 0.98,
                    "rmse": 0.37,
                    "bias": -0.15,
                    "mean_posterior_sd": 0.55,
                }
            },
        },
        "random": {
            "stimuli": [
                {"sequence_a": "HTHHTT", "sequence_b": "HTHTTT", "eig": 0.06},
                {"sequence_a": "TTTHHH", "sequence_b": "HTTHHT", "eig": 0.03},
            ],
            "mean_stimulus_eig": 0.045,
            "runs": [
                {
                    "repeat": 0,
                    "true_params": {"beta": 1.0},
                    "posterior_mean": {"beta": 4.0},
                    "posterior_sd": {"beta": 2.0},
                    "final_entropy_bits": 5.0,
                },
                {
                    "repeat": 1,
                    "true_params": {"beta": 8.0},
                    "posterior_mean": {"beta": 6.0},
                    "posterior_sd": {"beta": 2.5},
                    "final_entropy_bits": 5.4,
                },
            ],
            "summary": {
                "beta": {
                    "pearson_r": 0.55,
                    "rmse": 2.6,
                    "bias": 0.5,
                    "mean_posterior_sd": 2.25,
                }
            },
        },
    },
}

MODEL_COMPARISON = {
    "model_names": ["A", "B"],
    "n_repeats": 2,
    "n_stimuli": 1,
    "n_participants": 10,
    "points_per_dim": 3,
    "seed": 0,
    "pool_size": 20,
    "arms": {
        "eig": {
            "stimuli": [
                {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT", "eig": 0.4}
            ],
            "mean_stimulus_eig": 0.4,
            "runs": [
                {
                    "repeat": 0,
                    "generating_model": "A",
                    "true_params": {"x": 0.5},
                    "recovered_model": "A",
                    "recovered_correct": True,
                    "model_posterior": {"A": 0.9, "B": 0.1},
                }
            ],
            "accuracy": 1.0,
            "mean_true_posterior": 0.9,
            "confusion": {"A": {"A": 0.9, "B": 0.1}, "B": {"A": 0.2, "B": 0.8}},
        },
        "random": {
            "stimuli": [
                {"sequence_a": "HTHHTT", "sequence_b": "HTHTTT", "eig": 0.02}
            ],
            "mean_stimulus_eig": 0.02,
            "runs": [
                {
                    "repeat": 0,
                    "generating_model": "A",
                    "true_params": {"x": 0.5},
                    "recovered_model": "B",
                    "recovered_correct": False,
                    "model_posterior": {"A": 0.4, "B": 0.6},
                }
            ],
            "accuracy": 0.5,
            "mean_true_posterior": 0.6,
            "confusion": {"A": {"A": 0.4, "B": 0.6}, "B": {"A": 0.45, "B": 0.55}},
        },
    },
}


def test_selection_comparison_parameter_text_shows_both_arms_per_parameter():
    text = selection_comparison_parameter_text(PARAM_COMPARISON)
    assert "demo" in text
    assert "eig" in text and "random" in text
    assert "beta" in text
    # Both arms' correlations and rmse appear for the side-by-side comparison.
    assert "0.98" in text and "0.55" in text
    assert "0.37" in text and "2.6" in text
    # The chosen sets' informativeness is contrasted between arms.
    assert "0.31" in text and "0.045" in text  # mean stimulus EIG per arm


def test_selection_comparison_model_text_shows_accuracy_for_both_arms():
    text = selection_comparison_model_text(MODEL_COMPARISON)
    assert "eig" in text and "random" in text
    assert "1.00" in text and "0.50" in text  # accuracy per arm
    assert "0.900" in text and "0.600" in text  # mean posterior on true model
    assert "A" in text and "B" in text  # per-generating-model rows


def test_plot_selection_comparison_parameters_writes_figure(tmp_path):
    out = tmp_path / "param_comparison.png"
    plot_selection_comparison_parameters(PARAM_COMPARISON, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_selection_comparison_models_writes_figure(tmp_path):
    out = tmp_path / "model_comparison.png"
    plot_selection_comparison_models(MODEL_COMPARISON, out)
    assert out.exists() and out.stat().st_size > 0


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
