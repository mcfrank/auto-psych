"""PyMC adapter for Griffiths et al.'s four-motif stack automaton.

The regular hypothesis is the maximum-probability combination of a Viterbi
hidden-state path and one of four production methods: ordinary motif
continuation, mirror symmetry, complement symmetry, or duplication.  The
symmetry indicators are exact precomputed features; the first-half and full
sequence Viterbi probabilities remain functions of the inferred ``delta`` and
``alpha`` parameters.

The paper estimated this model on fixed-length sequences.  Because the Viterbi
construction omits a length-specific normalizer, this adapter rejects
cross-length comparisons rather than silently treating scores from different
normalizing constants as commensurable.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.raise_op import Assert

MAX_SEQ_LEN = 8
N_STATES = 6
_EVEN_STATES = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def _matrices(delta, alpha):
    a, a2, d = alpha, alpha**2, delta
    zero = pt.zeros_like(a)
    rows = pt.stack(
        [
            pt.stack([d, a, a2, zero, zero, a2]),
            pt.stack([a, d, a2, zero, zero, a2]),
            pt.stack([a, a, zero, d, zero, a2]),
            pt.stack([a, a, d, zero, zero, a2]),
            pt.stack([a, a, a2, zero, zero, d]),
            pt.stack([a, a, a2, zero, d, zero]),
        ]
    )
    transition = rows / rows.sum(axis=1, keepdims=True)
    init_raw = pt.stack([a, a, a2, zero, zero, a2])
    return init_raw / init_raw.sum(), transition


def _viterbi_log_probabilities(n, syms, init, transition):
    """Return per-trial log Viterbi probabilities for full and first half."""
    emissions = []
    for symbol in syms:
        symbol_float = pt.cast(symbol, "float64")
        emissions.append(
            symbol_float[:, None] * _EVEN_STATES[None, :]
            + (1.0 - symbol_float)[:, None]
            * (1.0 - _EVEN_STATES)[None, :]
        )

    best = init[None, :] * emissions[0]
    prefix_logs = [pt.log(pt.max(best, axis=1))]
    for emission in emissions[1:]:
        path_probabilities = best[:, :, None] * transition[None, :, :]
        best = pt.max(path_probabilities, axis=1) * emission
        prefix_logs.append(pt.log(pt.max(best, axis=1)))

    by_length = pt.stack(prefix_logs, axis=1)
    full_index = pt.cast(n - 1, "int64")[:, None]
    half_index = pt.cast((n - 1) // 2, "int64")[:, None]
    full = pt.take_along_axis(by_length, full_index, axis=1)[:, 0]
    half = pt.take_along_axis(by_length, half_index, axis=1)[:, 0]
    return full, half


def _log_p_regular(
    n,
    syms,
    mirror_symmetry,
    complement_symmetry,
    duplication,
    init,
    transition,
    method_weights,
):
    full_log_probability, half_log_probability = _viterbi_log_probabilities(
        n, syms, init, transition
    )
    impossible = pt.full_like(full_log_probability, -np.inf)
    component_logs = pt.stack(
        [
            pt.log(method_weights[0]) + full_log_probability,
            pt.switch(
                pt.eq(mirror_symmetry, 1),
                pt.log(method_weights[1]) + half_log_probability,
                impossible,
            ),
            pt.switch(
                pt.eq(complement_symmetry, 1),
                pt.log(method_weights[2]) + half_log_probability,
                impossible,
            ),
            pt.switch(
                pt.eq(duplication, 1),
                pt.log(method_weights[3]) + half_log_probability,
                impossible,
            ),
        ],
        axis=0,
    )
    return pt.max(component_logs, axis=0)


with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    syms_a = [
        pm.Data(f"sym{i}_a", np.zeros(1, dtype="int64"))
        for i in range(1, MAX_SEQ_LEN + 1)
    ]
    syms_b = [
        pm.Data(f"sym{i}_b", np.zeros(1, dtype="int64"))
        for i in range(1, MAX_SEQ_LEN + 1)
    ]
    mirror_symmetry_a = pm.Data(
        "mirror_symmetry_a", np.zeros(1, dtype="int64")
    )
    mirror_symmetry_b = pm.Data(
        "mirror_symmetry_b", np.zeros(1, dtype="int64")
    )
    complement_symmetry_a = pm.Data(
        "complement_symmetry_a", np.zeros(1, dtype="int64")
    )
    complement_symmetry_b = pm.Data(
        "complement_symmetry_b", np.zeros(1, dtype="int64")
    )
    duplication_a = pm.Data("duplication_a", np.zeros(1, dtype="int64"))
    duplication_b = pm.Data("duplication_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    checked_n_a = Assert(
        "motif_stack requires same-length alternatives on every trial"
    )(n_a, pt.all(pt.eq(n_a, n_b)))

    delta = pm.Uniform("delta", lower=0.01, upper=0.99)
    alpha = pm.Uniform("alpha", lower=0.01, upper=0.99)
    repetition_weight = pm.Uniform("repetition_weight", lower=0.01, upper=0.99)
    mirror_share = pm.Uniform("mirror_share", lower=0.01, upper=0.99)
    complement_share = pm.Uniform("complement_share", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    init, transition = _matrices(delta, alpha)
    remaining = 1.0 - repetition_weight
    mirror_weight = remaining * mirror_share
    remaining_after_mirror = remaining * (1.0 - mirror_share)
    complement_weight = remaining_after_mirror * complement_share
    duplication_weight = remaining_after_mirror * (1.0 - complement_share)
    method_weights = pt.stack(
        [
            repetition_weight,
            mirror_weight,
            complement_weight,
            duplication_weight,
        ]
    )
    log_regular_a = _log_p_regular(
        checked_n_a,
        syms_a,
        mirror_symmetry_a,
        complement_symmetry_a,
        duplication_a,
        init,
        transition,
        method_weights,
    )
    log_regular_b = _log_p_regular(
        n_b,
        syms_b,
        mirror_symmetry_b,
        complement_symmetry_b,
        duplication_b,
        init,
        transition,
        method_weights,
    )

    score_a = pt.cast(checked_n_a, "float64") * np.log(0.5) - log_regular_a
    score_b = pt.cast(n_b, "float64") * np.log(0.5) - log_regular_b
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
