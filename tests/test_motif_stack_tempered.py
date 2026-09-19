"""motif_stack_tempered: a temperature dial between Viterbi and full-marginal.

``motif_stack_tempered`` replaces each of ``motif_stack``'s two maxes (over
hidden HMM paths, and over the four production methods) with a temperature
softmax ``tau * logsumexp(v/tau)``.  As ``tau -> 0`` that operator is exactly
``max``; at ``tau = 1`` it is the log-sum (mixture).  So the family spans from
the Viterbi ``motif_stack`` to the fully-marginalising ``motif_stack_softmax``,
and an intermediate ``tau`` smooths the likelihood while keeping predictions
close to the hard-max model.

This module pins:

* the two endpoints — ``(path=0, method=0)`` reproduces ``motif_stack`` and
  ``(path=1, method=1)`` reproduces ``motif_stack_softmax``, to floating-point
  noise, over random parameters;
* monotonic fidelity — as ``tau`` shrinks toward 0 the predictions move
  monotonically back toward the Viterbi model;
* the PyMC adapter matches its pure-Python twin at the temperatures the adapter
  actually ships with (whatever they are), the same twin-anchoring discipline as
  the other two adapters;
* the container layout is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.model_loading import load_pymc_model, pm_data_inputs
from src.subjective_randomness.model_families import motif_stack as viterbi_twin
from src.subjective_randomness.model_families import motif_stack_softmax as softmax_twin
from src.subjective_randomness.model_families import motif_stack_tempered as tempered_twin
from src.subjective_randomness.pymc_model_families import (
    motif_stack_tempered as tempered_adapter,
)
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

MODEL_NAME = "motif_stack_tempered"

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

_ALL_SEQUENCES = sorted(
    {s[side] for s in STIMULI for side in ("sequence_a", "sequence_b")}
)


def _random_params(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {
        name: float(rng.uniform(low, high))
        for name, (low, high) in tempered_twin.PARAM_BOUNDS.items()
    }


# --- Endpoints: the family spans the two existing models -------------------


@pytest.mark.parametrize("seed", range(5))
def test_zero_temperature_reproduces_viterbi_motif_stack(seed):
    params = _random_params(seed)
    for seq in _ALL_SEQUENCES:
        got = tempered_twin.log_p_regular(
            seq, params, path_temperature=0.0, method_temperature=0.0
        )
        expected = viterbi_twin.log_p_regular(seq, params)
        assert abs(got - expected) < 1e-9, (seq, got, expected)


@pytest.mark.parametrize("seed", range(5))
def test_unit_temperature_reproduces_softmax_motif_stack(seed):
    params = _random_params(seed)
    for seq in _ALL_SEQUENCES:
        got = tempered_twin.log_p_regular(
            seq, params, path_temperature=1.0, method_temperature=1.0
        )
        expected = softmax_twin.log_p_regular(seq, params)
        assert abs(got - expected) < 1e-9, (seq, got, expected)


def test_predictions_move_monotonically_toward_viterbi_as_temperature_shrinks():
    """Smaller tau => predictions closer to the hard-max model, per stimulus."""
    params = dict(viterbi_twin.DEFAULT_PARAMS)
    viterbi_p = p_left_model_family("motif_stack", STIMULI, params)
    prev_gap = None
    for tau in (1.0, 0.5, 0.25, 0.1):
        tempered_p = np.array(
            [
                tempered_twin.predict_left(
                    s, params, path_temperature=tau, method_temperature=tau
                )
                for s in STIMULI
            ]
        )
        gap = float(np.mean(np.abs(tempered_p - viterbi_p)))
        if prev_gap is not None:
            assert gap <= prev_gap + 1e-9, (tau, gap, prev_gap)
        prev_gap = gap
    # And the smallest tau is genuinely close to Viterbi.
    assert prev_gap < 5e-3


# --- The adapter matches its twin at the shipped temperatures --------------


@pytest.mark.parametrize("draw", range(5))
def test_adapter_matches_twin_at_shipped_temperatures(draw):
    """The adapter bakes in its temperatures; the twin takes them as kwargs."""
    params = _random_params(draw + 100)
    from_pymc = p_left_fixed_params(MODEL_NAME, MODEL_DIR, STIMULI, params)
    from_twin = np.array(
        [
            tempered_twin.predict_left(
                s,
                params,
                path_temperature=tempered_adapter.PATH_TEMPERATURE,
                method_temperature=tempered_adapter.METHOD_TEMPERATURE,
            )
            for s in STIMULI
        ]
    )
    np.testing.assert_allclose(from_pymc, from_twin, atol=1e-8, rtol=0)
    assert from_twin.std() > 1e-3


def test_shipped_temperatures_are_in_the_open_unit_interval():
    """tau in (0, 1]: 0 would be the un-smoothed Viterbi model, >1 is undefined here."""
    assert 0.0 < tempered_adapter.PATH_TEMPERATURE <= 1.0
    assert 0.0 < tempered_adapter.METHOD_TEMPERATURE <= 1.0


def test_model_reads_the_same_container_layout():
    model = load_pymc_model(MODEL_NAME, MODEL_DIR)
    assert set(pm_data_inputs(model)) == EXPECTED_DATA_INPUTS
