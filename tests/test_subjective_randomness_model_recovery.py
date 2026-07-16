"""Tests for closed-ended model recovery in the subjective-randomness domain.

Fast tests cover the synthetic-data generator (fixed-parameter prior predictive,
no NUTS) and the confusion-matrix reshaping. The end-to-end recovery wiring,
which fits models via MCMC, is covered by the `slow`-marked integration test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.subjective_randomness.model_recovery as model_recovery
from src.subjective_randomness.model_recovery import (
    confusion_tidy_rows,
    default_generating_params,
    feature_rows,
    generate_responses,
    p_left_fixed_params,
    p_left_model_family,
    resolve_generating_params,
    run_closed_ended_recovery,
    run_recovery_from_config,
    write_responses_csv,
)
from src.subjective_randomness.features import featurize_stimulus

REPO_ROOT = Path(__file__).resolve().parent.parent
# The recovery GT/baseline registry: the original validated model set with
# pure-Python family twins. NOT the live project seed_models dir, which since
# the hero-run promotion holds the replicate winners (no family twins).
SEED_MODELS_DIR = REPO_ROOT / "src/subjective_randomness/pymc_model_families"

# Two stimuli the prototype-similarity model should rate very differently:
# a clearly "more random looking" alternating pair vs. a long-run pair.
STIMULI = [
    {"sequence_a": "HTHTHT", "sequence_b": "HHHHHH"},
    {"sequence_a": "HHHTTT", "sequence_b": "HTHTHT"},
]
PROTOTYPE_PARAMS = {
    "theta_alt": 0.65,
    "alt_weight": 0.55,
    "beta": 4.0,
    "side_bias": 0.0,
}


# ── feature_rows (public: consumed by holdout_recovery) ─────────────


def test_feature_rows_carries_sequences_features_and_dummy_response():
    rows = feature_rows(STIMULI)
    assert len(rows) == len(STIMULI)
    for stim, row in zip(STIMULI, rows):
        assert row["sequence_a"] == stim["sequence_a"]
        assert row["sequence_b"] == stim["sequence_b"]
        # The dummy observed-response column exists (PyMC models declare it as
        # a pm.Data input) but carries no information.
        assert row["chose_left"] == 0
        # Every derived feature matches the canonical featurizer exactly.
        expected = featurize_stimulus(stim["sequence_a"], stim["sequence_b"])
        assert {k: row[k] for k in expected} == expected


def test_feature_rows_rejects_empty_stimuli():
    with pytest.raises(ValueError, match="No stimuli"):
        feature_rows([])


def test_p_left_fixed_params_is_deterministic_and_valid():
    a = p_left_fixed_params(
        "prototype_similarity", SEED_MODELS_DIR, STIMULI, PROTOTYPE_PARAMS, seed=0
    )
    b = p_left_fixed_params(
        "prototype_similarity", SEED_MODELS_DIR, STIMULI, PROTOTYPE_PARAMS, seed=999
    )
    # Fixed params -> p_left is a deterministic function of the stimuli; the RNG
    # seed must not move it (it only governs the later Bernoulli draws).
    assert np.allclose(a, b)
    assert a.shape == (len(STIMULI),)
    assert np.all((a > 0.0) & (a < 1.0))
    # Stimulus 0 favors side A (alternating); stimulus 1 favors side B (alternating).
    assert a[0] > 0.5
    assert a[1] < 0.5


def test_generate_responses_shape_and_columns():
    rows = generate_responses(
        "prototype_similarity",
        SEED_MODELS_DIR,
        STIMULI,
        PROTOTYPE_PARAMS,
        n_participants=5,
        seed=1,
    )
    assert len(rows) == 5 * len(STIMULI)
    cols = set(rows[0])
    assert {"participant_id", "trial_index", "sequence_a", "sequence_b",
            "chose_left", "generating_model"} <= cols
    # Full feature set must be present so any seed model can be fit on these rows.
    assert {"imbalance_a", "p_alts_a", "n_a", "h_a", "alts_a",
            "max_run_norm_a", "periodicity_a"} <= cols
    assert all(r["chose_left"] in (0, 1) for r in rows)
    assert all(r["generating_model"] == "prototype_similarity" for r in rows)


def test_generate_responses_is_seed_deterministic():
    kw = dict(n_participants=8, seed=7)
    rows_a = generate_responses(
        "prototype_similarity", SEED_MODELS_DIR, STIMULI, PROTOTYPE_PARAMS, **kw
    )
    rows_b = generate_responses(
        "prototype_similarity", SEED_MODELS_DIR, STIMULI, PROTOTYPE_PARAMS, **kw
    )
    assert [r["chose_left"] for r in rows_a] == [r["chose_left"] for r in rows_b]


def test_generate_responses_choice_rate_tracks_p_left():
    p_left = p_left_fixed_params(
        "prototype_similarity", SEED_MODELS_DIR, STIMULI, PROTOTYPE_PARAMS, seed=0
    )
    rows = generate_responses(
        "prototype_similarity",
        SEED_MODELS_DIR,
        STIMULI,
        PROTOTYPE_PARAMS,
        n_participants=400,
        seed=3,
    )
    for stim_idx, stim in enumerate(STIMULI):
        chose = [
            r["chose_left"]
            for r in rows
            if r["sequence_a"] == stim["sequence_a"]
            and r["sequence_b"] == stim["sequence_b"]
        ]
        assert abs(np.mean(chose) - p_left[stim_idx]) < 0.08


def test_generate_responses_model_family_backend_tracks_predict_left():
    # The pure-Python family generator should reproduce predict_left choice rates,
    # giving a harder recovery test (generator's functional form differs from the
    # PyMC fit of the same name).
    import importlib

    family = importlib.import_module(
        "src.subjective_randomness.model_families.prototype_similarity"
    )
    expected = [family.predict_left(stim, PROTOTYPE_PARAMS) for stim in STIMULI]

    rows = generate_responses(
        "prototype_similarity",
        SEED_MODELS_DIR,
        STIMULI,
        PROTOTYPE_PARAMS,
        n_participants=400,
        seed=2,
        generator="model_family",
    )
    cols = set(rows[0])
    assert {"imbalance_a", "p_alts_a", "n_a", "chose_left", "generating_model"} <= cols
    for stim_idx, stim in enumerate(STIMULI):
        chose = [
            r["chose_left"]
            for r in rows
            if r["sequence_a"] == stim["sequence_a"]
            and r["sequence_b"] == stim["sequence_b"]
        ]
        assert abs(np.mean(chose) - expected[stim_idx]) < 0.08


def test_generate_responses_rejects_unknown_generator():
    with pytest.raises(ValueError):
        generate_responses(
            "prototype_similarity",
            SEED_MODELS_DIR,
            STIMULI,
            PROTOTYPE_PARAMS,
            n_participants=2,
            seed=0,
            generator="nonsense",
        )


def test_generate_responses_rejects_zero_participants():
    with pytest.raises(ValueError, match="must be >= 1"):
        generate_responses(
            "prototype_similarity",
            SEED_MODELS_DIR,
            STIMULI,
            PROTOTYPE_PARAMS,
            n_participants=0,
            seed=0,
        )


def test_generate_responses_rejects_empty_stimuli():
    with pytest.raises(ValueError, match="No stimuli"):
        generate_responses(
            "prototype_similarity",
            SEED_MODELS_DIR,
            [],
            PROTOTYPE_PARAMS,
            n_participants=2,
            seed=0,
        )


def test_p_left_model_family_rejects_mismatched_params():
    # Wrong/incomplete params for the pure-Python family must fail loudly rather
    # than silently fall back to the family's defaults.
    with pytest.raises(ValueError, match="exactly"):
        p_left_model_family(
            "prototype_similarity", STIMULI, {"theta_alt": 0.65}  # missing the rest
        )


def test_write_responses_csv_rejects_empty_rows(tmp_path):
    with pytest.raises(ValueError, match="No response rows"):
        write_responses_csv([], tmp_path / "responses.csv")


def test_run_recovery_from_config_merges_overrides_and_delegates(tmp_path, monkeypatch):
    # The config-orchestration entry point should resolve paths, merge CLI
    # overrides over the config's fit settings, and delegate to the recovery
    # runner. Stub the runner so we exercise the plumbing without MCMC.
    stimuli_path = tmp_path / "stimuli.json"
    stimuli_path.write_text(
        json.dumps([{"sequence_a": "HTHT", "sequence_b": "HHHH"}]), encoding="utf-8"
    )
    config = {
        "seed_models_dir": str(SEED_MODELS_DIR),
        "stimuli_path": str(stimuli_path),
        "n_participants": 30,
        "seed": 1,
        "generating_models": ["prototype_similarity"],
        "fit": {"draws": 500, "tune": 500, "chains": 2},
    }

    captured = {}

    def fake_runner(stimuli, seed_models_dir, **kwargs):
        captured["stimuli"] = stimuli
        captured["seed_models_dir"] = seed_models_dir
        captured["kwargs"] = kwargs
        return {"generating": []}

    monkeypatch.setattr(model_recovery, "run_closed_ended_recovery", fake_runner)

    result = run_recovery_from_config(
        config,
        tmp_path / "config.yaml",
        tmp_path / "runs",
        n_participants_override=12,
        fit_overrides={"draws": 100},
        seed_override=7,
        generator_override="model_family",
    )

    assert result == {"generating": []}
    assert len(captured["stimuli"]) == 1
    kwargs = captured["kwargs"]
    assert kwargs["n_participants"] == 12  # CLI override wins over config's 30
    assert kwargs["seed"] == 7  # CLI override wins over config's 1
    assert kwargs["generator"] == "model_family"
    assert kwargs["fit_kwargs"] == {"draws": 100, "tune": 500, "chains": 2}  # merged
    assert set(kwargs["generating_params"]) == {"prototype_similarity"}


def test_generate_responses_rejects_incomplete_params():
    with pytest.raises(ValueError):
        generate_responses(
            "prototype_similarity",
            SEED_MODELS_DIR,
            STIMULI,
            {"theta_alt": 0.65},  # missing alt_weight, beta, side_bias
            n_participants=2,
            seed=0,
        )


# ── default generating params ───────────────────────────────────────


def test_default_generating_params_covers_every_seed_model():
    params = default_generating_params(SEED_MODELS_DIR)
    assert set(params) == {
        "prototype_similarity",
        "encoding_compressibility",
        "bayesian_diagnosticity",
        "window_typicality",
    }
    # Each default must name exactly the PyMC model's free parameters, so it can
    # drive the fixed-parameter generator without a loud failure.
    assert set(params["prototype_similarity"]) == {
        "theta_alt",
        "alt_weight",
        "beta",
        "side_bias",
    }
    assert set(params["bayesian_diagnosticity"]) == {
        "delta",
        "alpha",
        "bias_share",
        "beta",
        "side_bias",
    }
    assert set(params["window_typicality"]) == {
        "window",
        "over_alt_penalty",
        "beta",
        "side_bias",
    }


# ── config -> generating params resolution ──────────────────────────


def test_resolve_generating_params_none_uses_all_seed_defaults():
    resolved = resolve_generating_params(None, SEED_MODELS_DIR)
    assert resolved == default_generating_params(SEED_MODELS_DIR)


def test_resolve_generating_params_list_selects_subset_with_defaults():
    resolved = resolve_generating_params(["prototype_similarity"], SEED_MODELS_DIR)
    assert set(resolved) == {"prototype_similarity"}
    assert resolved["prototype_similarity"] == (
        default_generating_params(SEED_MODELS_DIR)["prototype_similarity"]
    )


def test_resolve_generating_params_rejects_invalid_type():
    # Anything that isn't None / a list / a name->params mapping must fail loudly.
    with pytest.raises(TypeError, match="must be null, a list"):
        resolve_generating_params(42, SEED_MODELS_DIR)


def test_resolve_generating_params_dict_overrides_and_fills_nulls():
    spec = {
        "prototype_similarity": {
            "theta_alt": 0.5,
            "alt_weight": 0.5,
            "beta": 6.0,
            "side_bias": 0.1,
        },
        "bayesian_diagnosticity": None,  # null -> fall back to family defaults
    }
    resolved = resolve_generating_params(spec, SEED_MODELS_DIR)
    assert set(resolved) == {"prototype_similarity", "bayesian_diagnosticity"}
    assert resolved["prototype_similarity"]["beta"] == 6.0
    assert resolved["bayesian_diagnosticity"] == (
        default_generating_params(SEED_MODELS_DIR)["bayesian_diagnosticity"]
    )


# ── confusion-matrix reshaping ──────────────────────────────────────

# A synthetic recovery result (shape produced by run_closed_ended_recovery),
# avoiding any MCMC so this stays a fast test.
RECOVERY_RESULT = {
    "seed_models": ["prototype_similarity", "encoding_compressibility"],
    "n_participants": 30,
    "n_stimuli": 12,
    "generating": [
        {
            "generating_model": "prototype_similarity",
            "best_model": "prototype_similarity",
            "recovered_correct": True,
            "posteriors": {
                "prototype_similarity": 0.7,
                "encoding_compressibility": 0.3,
            },
            "elpd_loo": {
                "prototype_similarity": -40.0,
                "encoding_compressibility": -45.0,
            },
        },
        {
            "generating_model": "encoding_compressibility",
            "best_model": "prototype_similarity",
            "recovered_correct": False,
            "posteriors": {
                "prototype_similarity": 0.55,
                "encoding_compressibility": 0.45,
            },
            "elpd_loo": {
                "prototype_similarity": -41.0,
                "encoding_compressibility": -41.5,
            },
        },
    ],
}


def test_confusion_tidy_rows_one_row_per_cell():
    rows = confusion_tidy_rows(RECOVERY_RESULT)
    assert len(rows) == 2 * 2  # generating x recovered

    diag = next(
        r
        for r in rows
        if r["generating_model"] == "prototype_similarity"
        and r["recovered_model"] == "prototype_similarity"
    )
    assert diag["posterior"] == 0.7
    assert diag["is_true_model"] is True
    assert diag["is_best_model"] is True

    off = next(
        r
        for r in rows
        if r["generating_model"] == "encoding_compressibility"
        and r["recovered_model"] == "prototype_similarity"
    )
    assert off["is_true_model"] is False
    assert off["is_best_model"] is True  # recovered the wrong model as best


# ── end-to-end recovery wiring (MCMC) ───────────────────────────────


@pytest.mark.slow
def test_run_closed_ended_recovery_assembles_confusion(tmp_path):
    # Recover a single generating model against the closed seed set. Uses tiny
    # MCMC settings — this pins the wiring (generate -> fit closed set -> confusion),
    # not the statistical recovery quality (which the real experiment measures).
    result = run_closed_ended_recovery(
        STIMULI * 4,  # a few stimuli so LOO has something to chew on
        SEED_MODELS_DIR,
        generating_params={"prototype_similarity": PROTOTYPE_PARAMS},
        n_participants=12,
        results_root=tmp_path / "recovery",
        fit_kwargs={"draws": 50, "tune": 50, "chains": 2},
        seed=0,
    )

    seed_set = set(result["seed_models"])
    assert seed_set == {
        "prototype_similarity",
        "encoding_compressibility",
        "bayesian_diagnosticity",
        "window_typicality",
    }
    assert len(result["generating"]) == 1

    record = result["generating"][0]
    assert record["generating_model"] == "prototype_similarity"
    # Every seed model is scored; the posterior is a distribution over them.
    assert set(record["posteriors"]) == seed_set
    # Posteriors are stored rounded to 6 dp; the sum can drift by a few ULPs of
    # that rounding across the seed set, so allow a 1e-5 tolerance.
    assert abs(sum(record["posteriors"].values()) - 1.0) < 1e-5
    assert record["best_model"] in seed_set
    assert record["recovered_correct"] == (
        record["best_model"] == "prototype_similarity"
    )

    # The PSIS-LOO comparison table (elpd_diff/dse per model) is carried through
    # so downstream analysis can judge whether a recovery is statistically clear.
    assert set(record["comparison"]) == seed_set
    top = record["comparison"]["prototype_similarity"]
    assert {"rank", "elpd_diff", "dse", "weight"} <= set(top)

    # The per-model inner-loop artifacts were written to disk.
    assert (tmp_path / "recovery" / "prototype_similarity" / "responses.csv").exists()
