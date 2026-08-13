"""motif_stack scores each UNIQUE sequence once, not once per trial row.

The adapter's cost is dominated by an unrolled 8-step Viterbi recursion. Run
per trial row that recursion is recomputed for every repetition of the same
stimulus: a 30-participant x 32-pair design has 960 rows but at most 64 distinct
sequences, so ~94% of the work was redundant (measured 2026-08-13: 6.7 ms per
logp+grad evaluation at 960 rows vs 0.47 ms at 32).

This module pins the restructure:

* the model reads a unique-sequence table plus per-trial gather indices;
* ``p_left`` stays a per-trial ``pm.Deterministic`` of shape ``(T,)`` and the
  observed Bernoulli stays per-trial, because ELPD-LOO granularity and
  ``predict_p_left_draws`` depend on it;
* the likelihood is UNCHANGED — it still agrees with the pure-Python twin;
* the same-length restriction is enforced loudly in Python at data-preparation
  time instead of by an ``Assert`` embedded in the logp graph.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from src.models.pymc_inference import (
    load_pymc_model,
    make_stim_data,
    observed_response_data,
    pm_data_inputs,
)
from src.subjective_randomness.model_families import motif_stack as twin
from src.subjective_randomness.model_recovery import (
    p_left_fixed_params,
    p_left_model_family,
)

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "subjective_randomness"
    / "pymc_model_families"
)
SEED_MODELS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "pipelines"
    / "outer_loop"
    / "projects"
    / "subjective_randomness"
    / "seed_models"
)

# The unique-sequence table plus per-trial gather indices. The old layout named
# one container per trial-aligned symbol column (sym1_a..sym8_b, n_a/n_b, and
# the three symmetry flags per side) — 27 containers, all recomputed per row.
EXPECTED_DATA_INPUTS = {
    "seq_len",
    "emission_mask",
    "mirror_symmetry",
    "complement_symmetry",
    "duplication",
    "idx_a",
    "idx_b",
    "chose_left",
}

RETIRED_DATA_INPUTS = {
    "n_a",
    "n_b",
    *{f"sym{i}_{side}" for i in range(1, 9) for side in ("a", "b")},
    *{
        f"{flag}_{side}"
        for flag in ("mirror_symmetry", "complement_symmetry", "duplication")
        for side in ("a", "b")
    },
}


def _stimulus_bank() -> list[dict[str, str]]:
    """Same-length pairs over lengths 4-8, with every memory flag firing.

    Sequences are enumerated exhaustively for each length (16-256 strings) and
    paired up, so the bank spans symmetric and asymmetric sequences alike rather
    than only the handful a hand-written list would cover.
    """
    stimuli: list[dict[str, str]] = []
    for length in range(4, 9):
        words = [
            "".join("HT"[(i >> bit) & 1] for bit in range(length))
            for i in range(2**length)
        ]
        # Pair each sequence with the next one, wrapping, so every sequence
        # appears on both sides across the bank.
        step = max(1, len(words) // 24)
        chosen = words[::step]
        for i, seq_a in enumerate(chosen):
            stimuli.append(
                {"sequence_a": seq_a, "sequence_b": chosen[(i + 1) % len(chosen)]}
            )
    return [s for s in stimuli if s["sequence_a"] != s["sequence_b"]]


STIMULI = _stimulus_bank()


def test_stimulus_bank_is_large_and_exercises_every_memory_flag():
    """Guard the guard: an equivalence test over degenerate stimuli proves little."""
    assert len(STIMULI) >= 100
    assert {len(s["sequence_a"]) for s in STIMULI} == {4, 5, 6, 7, 8}
    fired = {"mirror": 0, "complement": 0, "duplication": 0}
    for stim in STIMULI:
        for side in ("sequence_a", "sequence_b"):
            for flag, matched in twin._memory_patterns(stim[side]).items():
                fired[flag] += int(matched)
    assert all(count > 0 for count in fired.values()), fired


# --- Integration: the likelihood is unchanged ------------------------------


@pytest.mark.parametrize("draw", range(6))
def test_deduped_adapter_matches_the_pure_python_twin(draw):
    """Numeric equivalence at 1e-8 over random parameter draws x 100+ stimuli.

    The pure-Python twin is the anchor; the restructure is purely computational,
    so agreement must hold to floating-point noise for ANY parameter vector, not
    just the published defaults.
    """
    rng = np.random.default_rng(draw)
    params = {
        name: float(rng.uniform(low, high))
        for name, (low, high) in twin.PARAM_BOUNDS.items()
    }
    from_pymc = p_left_fixed_params("motif_stack", MODEL_DIR, STIMULI, params)
    from_twin = p_left_model_family("motif_stack", STIMULI, params)
    np.testing.assert_allclose(from_pymc, from_twin, atol=1e-8, rtol=0)
    # Not vacuous: the predictions must actually vary across stimuli.
    assert from_twin.std() > 1e-3


def test_default_params_match_the_twin():
    from_pymc = p_left_fixed_params(
        "motif_stack", MODEL_DIR, STIMULI, dict(twin.DEFAULT_PARAMS)
    )
    from_twin = p_left_model_family("motif_stack", STIMULI, dict(twin.DEFAULT_PARAMS))
    np.testing.assert_allclose(from_pymc, from_twin, atol=1e-8, rtol=0)


# --- The new data layout ---------------------------------------------------


def test_model_reads_a_unique_sequence_table_and_gather_indices():
    model = load_pymc_model("motif_stack", MODEL_DIR)
    assert set(pm_data_inputs(model)) == EXPECTED_DATA_INPUTS
    assert observed_response_data(model) == "chose_left"


def test_retired_per_trial_columns_are_gone():
    model = load_pymc_model("motif_stack", MODEL_DIR)
    assert RETIRED_DATA_INPUTS.isdisjoint(set(pm_data_inputs(model)))


def test_viterbi_runs_once_per_unique_sequence_not_once_per_trial():
    """30 participants x the same 32 pairs = 960 rows but <= 64 sequences."""
    model = load_pymc_model("motif_stack", MODEL_DIR)
    pairs = STIMULI[:32]
    rows = [{**pair, "chose_left": 0} for _ in range(30) for pair in pairs]
    stim_data = make_stim_data(model, rows)

    n_trials = len(rows)
    n_unique = len(stim_data["seq_len"])
    assert n_trials == 960
    assert n_unique <= 64
    # Per-trial arrays stay per-trial; the scored batch shrinks to the unique set.
    assert stim_data["idx_a"].shape == (n_trials,)
    assert stim_data["idx_b"].shape == (n_trials,)
    assert stim_data["chose_left"].shape == (n_trials,)
    assert stim_data["emission_mask"].shape == (twin_max_len(), n_unique, 6)
    assert stim_data["mirror_symmetry"].shape == (n_unique,)
    # The indices really point at the sequences they came from.
    unique_seqs = _unique_sequences(rows)
    for row, index_a, index_b in zip(rows, stim_data["idx_a"], stim_data["idx_b"]):
        assert unique_seqs[index_a] == row["sequence_a"]
        assert unique_seqs[index_b] == row["sequence_b"]


def twin_max_len() -> int:
    module = importlib.import_module(
        "src.subjective_randomness.pymc_model_families.motif_stack"
    )
    return module.MAX_SEQ_LEN


def _unique_sequences(rows) -> list[str]:
    module = importlib.import_module(
        "src.subjective_randomness.pymc_model_families.motif_stack"
    )
    return module.unique_sequences(rows)


def test_p_left_and_observed_response_stay_per_trial():
    """ELPD-LOO granularity and predict_p_left_draws need one p_left per trial."""
    import pymc as pm

    model = load_pymc_model("motif_stack", MODEL_DIR)
    rows = [{**pair, "chose_left": 0} for pair in STIMULI[:12]]
    with model:
        pm.set_data(make_stim_data(model, rows))
        prior = pm.sample_prior_predictive(
            draws=3, var_names=["p_left", "response"], random_seed=0
        )
    assert prior.prior["p_left"].values.shape[-1] == len(rows)
    assert prior.prior_predictive["response"].values.shape[-1] == len(rows)


# --- Item 4: the Assert leaves the graph, Python validates loudly ----------


def test_mixed_length_pair_raises_at_data_preparation_time():
    model = load_pymc_model("motif_stack", MODEL_DIR)
    rows = [{"sequence_a": "HT", "sequence_b": "HTHT", "chose_left": 0}]
    with pytest.raises(ValueError, match="same-length"):
        make_stim_data(model, rows)


def test_mixed_length_message_names_the_model_and_the_offending_row():
    model = load_pymc_model("motif_stack", MODEL_DIR)
    rows = [
        {"sequence_a": "HTHT", "sequence_b": "HTHT", "chose_left": 0},
        {"sequence_a": "HTH", "sequence_b": "HTHT", "chose_left": 1},
    ]
    with pytest.raises(ValueError) as excinfo:
        make_stim_data(model, rows)
    message = str(excinfo.value)
    assert "motif_stack" in message
    assert "same-length" in message
    assert "row 1" in message


def test_no_data_assert_remains_in_the_logp_graph():
    """The same-length check must not cost a node in every logp evaluation.

    PyMC's own parameter-support guards (``Check{0 <= p <= 1}`` and friends) are
    also ``CheckAndRaise`` ops and belong in the graph — they validate
    *parameters*, which change every draw. What must be gone is the check on the
    *data*, which cannot change during a fit yet was re-evaluated on every one of
    the millions of logp calls a fit makes.
    """
    from pytensor.graph.traversal import ancestors
    from pytensor.raise_op import Assert, CheckAndRaise

    model = load_pymc_model("motif_stack", MODEL_DIR)
    check_ops = [
        var.owner.op
        for var in ancestors([model.logp()])
        if var.owner is not None and isinstance(var.owner.op, CheckAndRaise)
    ]
    assert [op for op in check_ops if isinstance(op, Assert)] == []
    assert [op for op in check_ops if "same-length" in str(op.msg)] == []


# --- The prediction paths pass rows without real responses ----------------


def test_rows_without_chose_left_are_accepted_with_dummy_zeros():
    """EIG design pools and predict_p_left_draws pass stimuli, not responses."""
    model = load_pymc_model("motif_stack", MODEL_DIR)
    rows = [{"sequence_a": s["sequence_a"], "sequence_b": s["sequence_b"]} for s in STIMULI[:5]]
    stim_data = make_stim_data(model, rows)
    assert stim_data["chose_left"].tolist() == [0, 0, 0, 0, 0]


def test_partially_present_chose_left_raises():
    """Half-populated responses are a caller bug, not something to paper over."""
    model = load_pymc_model("motif_stack", MODEL_DIR)
    rows = [
        {"sequence_a": "HTHT", "sequence_b": "HHTT", "chose_left": 1},
        {"sequence_a": "HTHT", "sequence_b": "TTHH"},
    ]
    with pytest.raises(ValueError, match="chose_left"):
        make_stim_data(model, rows)


def test_rows_missing_raw_sequences_raise():
    model = load_pymc_model("motif_stack", MODEL_DIR)
    with pytest.raises(ValueError, match="sequence_a"):
        make_stim_data(model, [{"n_a": 8, "n_b": 8, "chose_left": 0}])


def test_sequence_longer_than_the_supported_maximum_raises():
    model = load_pymc_model("motif_stack", MODEL_DIR)
    rows = [{"sequence_a": "H" * 9, "sequence_b": "T" * 9, "chose_left": 0}]
    with pytest.raises(ValueError, match="longer than"):
        make_stim_data(model, rows)


# --- motif_stack must survive the EIG design-pool screener ---------------


def test_motif_stack_is_not_dropped_by_the_eig_screener():
    """`_screen_usable_models` is the one place a model may be omitted from the
    hypothesis set. A `prepare_observed` hook that could not bind a design-pool
    row would get motif_stack excluded from EIG with only a printed `[drop]`
    line — quietly renormalizing the design over the remaining models."""
    from src.pipelines.outer_loop.eig import _feature_row, _screen_usable_models
    from src.subjective_randomness.features import featurize_stimulus

    probe_row = _feature_row(
        {"sequence_a": "HHTHTTHT", "sequence_b": "HTHTHTHT"}, featurize_stimulus
    )
    usable = _screen_usable_models(["motif_stack"], MODEL_DIR, probe_row)
    assert usable == ["motif_stack"]


# --- End-to-end: a real (tiny) MCMC fit ----------------------------------


@pytest.mark.slow
def test_smoke_fit_and_posterior_prediction(tmp_path):
    """A real NUTS fit on the new layout, through the CSV path, end to end.

    Cheap settings (50/50, 2 chains) — this checks that the restructured graph
    samples, that the model-declared target_accept=0.9 keeps it clean, and that
    posterior prediction still returns one probability per stimulus.
    """
    import csv

    from src.models.pymc_inference import fit_model
    from src.subjective_randomness.features import featurize_responses_csv

    rng = np.random.default_rng(0)
    pairs = [(s["sequence_a"], s["sequence_b"]) for s in STIMULI if len(s["sequence_a"]) == 8][:16]
    assert pairs, "need length-8 pairs for the smoke fit"

    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["participant_id", "trial_index", "sequence_a", "sequence_b", "chose_left"],
        )
        writer.writeheader()
        for participant in range(4):
            for trial, (seq_a, seq_b) in enumerate(pairs):
                p_left = twin.predict_left({"sequence_a": seq_a, "sequence_b": seq_b})
                writer.writerow(
                    {
                        "participant_id": f"p{participant}",
                        "trial_index": trial,
                        "sequence_a": seq_a,
                        "sequence_b": seq_b,
                        "chose_left": int(rng.random() < p_left),
                    }
                )
    responses = tmp_path / "responses.csv"
    n_rows = featurize_responses_csv(raw, responses)

    fitted = fit_model(
        "motif_stack",
        MODEL_DIR,
        responses,
        draws=50,
        tune=50,
        chains=2,
        cores=2,
    )
    # The fit is pointwise per trial, which is what ELPD-LOO comparison needs.
    assert fitted.idata.log_likelihood["response"].values.shape[-1] == n_rows
    n_divergences = int(fitted.idata.sample_stats["diverging"].values.sum())
    assert n_divergences == 0, f"{n_divergences} divergences at target_accept=0.9"

    stim_rows = [{**pair, "chose_left": 0} for pair in STIMULI[:6]]
    draws = fitted.predict_p_left_draws(make_stim_data(fitted.model, stim_rows))
    assert draws.shape == (100, len(stim_rows))
    assert np.all((draws > 0.0) & (draws < 1.0))


# --- The two on-disk copies must not drift -------------------------------


def test_seed_model_copy_is_byte_identical():
    registry = (MODEL_DIR / "motif_stack.py").read_bytes()
    staged = (SEED_MODELS_DIR / "motif_stack.py").read_bytes()
    assert registry == staged
