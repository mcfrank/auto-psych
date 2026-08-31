"""motif_stack_softmax: the softmax (marginalising) rewrite of motif_stack.

``motif_stack`` scores a sequence's regularity by the SINGLE most probable
combination of a hidden path and a production method — a double max: Viterbi
(max-product) over the HMM's hidden paths, and an argmax over the four
production methods.  Both maxes put non-differentiable ridges into the
log-likelihood wherever the argmax switches, which is exactly why the Viterbi
adapter has to lower ``target_accept`` and still takes long NUTS trajectories.

The softmax variant replaces both maxes with their log-sum-exp (temperature-1
softmax) counterparts:

* the Viterbi max-product recursion becomes the forward sum-product recursion
  (marginalise over hidden paths);
* the max over production methods becomes a mixture over them (marginalise over
  the method M, weighted by ``P(M)``).

``log P(x | regular)`` is then the exact marginal likelihood — a smooth function
of the parameters — instead of a max.  This module pins the contract of that
rewrite:

* the softmax PyMC adapter agrees with its pure-Python twin to 1e-8, the same
  twin-anchoring discipline the Viterbi adapter is held to;
* the rewrite actually changed the model — its predictions differ from the
  Viterbi model on the same stimuli and parameters;
* softmax >= max: a sum dominates a max, so the softmax regular probability is
  never below the Viterbi one and its randomness score is never above.  This is
  the defining inequality of the relaxation.
* the container layout is unchanged, so the same ``prepare_observed`` /
  unique-sequence-table plumbing carries over untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.pymc_inference import load_pymc_model, pm_data_inputs
from src.subjective_randomness.model_families import motif_stack as viterbi_twin
from src.subjective_randomness.model_families import motif_stack_softmax as softmax_twin
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

MODEL_NAME = "motif_stack_softmax"

# The Viterbi adapter's own container layout — the softmax rewrite touches only
# the likelihood arithmetic, so its data plumbing must be byte-for-byte the same.
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


def _stimulus_bank() -> list[dict[str, str]]:
    """Same-length pairs over lengths 4-8, spanning every memory flag.

    Enumerates all sequences of each length, strides to a manageable subset, and
    pairs each with the next (wrapping), so both symmetric and asymmetric
    sequences appear on both sides.
    """
    stimuli: list[dict[str, str]] = []
    for length in range(4, 9):
        words = [
            "".join("HT"[(i >> bit) & 1] for bit in range(length))
            for i in range(2**length)
        ]
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
            for flag, matched in softmax_twin._memory_patterns(stim[side]).items():
                fired[flag] += int(matched)
    assert all(count > 0 for count in fired.values()), fired


# --- The softmax adapter matches its pure-Python twin ----------------------


@pytest.mark.parametrize("draw", range(6))
def test_softmax_adapter_matches_the_pure_python_twin(draw):
    """Numeric equivalence at 1e-8 over random parameter draws x 100+ stimuli."""
    rng = np.random.default_rng(draw)
    params = {
        name: float(rng.uniform(low, high))
        for name, (low, high) in softmax_twin.PARAM_BOUNDS.items()
    }
    from_pymc = p_left_fixed_params(MODEL_NAME, MODEL_DIR, STIMULI, params)
    from_twin = p_left_model_family(MODEL_NAME, STIMULI, params)
    np.testing.assert_allclose(from_pymc, from_twin, atol=1e-8, rtol=0)
    assert from_twin.std() > 1e-3


def test_default_params_match_the_twin():
    from_pymc = p_left_fixed_params(
        MODEL_NAME, MODEL_DIR, STIMULI, dict(softmax_twin.DEFAULT_PARAMS)
    )
    from_twin = p_left_model_family(
        MODEL_NAME, STIMULI, dict(softmax_twin.DEFAULT_PARAMS)
    )
    np.testing.assert_allclose(from_pymc, from_twin, atol=1e-8, rtol=0)


# --- The rewrite really is a different model -------------------------------


def test_softmax_predictions_differ_from_viterbi():
    """A softmax rewrite that predicted identically to the max would be a no-op."""
    params = dict(viterbi_twin.DEFAULT_PARAMS)
    softmax_p = p_left_model_family(MODEL_NAME, STIMULI, params)
    viterbi_p = p_left_model_family("motif_stack", STIMULI, params)
    assert np.max(np.abs(softmax_p - viterbi_p)) > 1e-2


def test_softmax_regular_probability_never_below_viterbi():
    """softmax >= max, per sequence: sum over paths/methods dominates their max.

    Checked on ``log_p_regular`` directly, over both sides of every stimulus, so
    the inequality is exercised on symmetric and asymmetric sequences alike.
    """
    params = dict(viterbi_twin.DEFAULT_PARAMS)
    sequences = sorted(
        {s[side] for s in STIMULI for side in ("sequence_a", "sequence_b")}
    )
    softmax_logs = np.array(
        [softmax_twin.log_p_regular(seq, params) for seq in sequences]
    )
    viterbi_logs = np.array(
        [viterbi_twin.log_p_regular(seq, params) for seq in sequences]
    )
    assert np.all(softmax_logs >= viterbi_logs - 1e-9)
    # Not vacuous: the relaxation must strictly lift at least some sequences.
    assert np.max(softmax_logs - viterbi_logs) > 1e-3


# --- Same data plumbing as the Viterbi adapter -----------------------------


def test_model_reads_the_same_container_layout_as_viterbi():
    model = load_pymc_model(MODEL_NAME, MODEL_DIR)
    assert set(pm_data_inputs(model)) == EXPECTED_DATA_INPUTS
