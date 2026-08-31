"""PyMC adapter for the temperature-smoothed four-motif stack automaton.

This adapter interpolates between the Viterbi ``motif_stack`` and the fully
marginalising ``motif_stack_softmax`` via two module-level temperatures:

* ``PATH_TEMPERATURE`` softens the max over hidden HMM paths;
* ``METHOD_TEMPERATURE`` softens the max over the four production methods.

Each max ``m`` is replaced by the temperature softmax ``tau * logsumexp(v/tau)``,
which is ``max(v)`` as ``tau -> 0`` and the log-sum (mixture) at ``tau = 1``.

Implementation notes
--------------------
* The hidden-path recursion runs in probability space, like the softmax adapter,
  but with the init/transition factors raised to ``1/PATH_TEMPERATURE`` and the
  per-prefix score scaled by ``PATH_TEMPERATURE`` — i.e. ``tau * log Z`` of the
  model whose factors are tempered by ``1/tau``.  This is exact at ``tau = 1``
  (ordinary forward) and approaches the Viterbi max as ``tau -> 0``.  It is only
  numerically safe away from very small ``tau`` (the powered products underflow
  as ``tau -> 0``); the checked-in temperatures stay well inside the safe range,
  and the pure-Python twin (which runs in log space) is the reference the
  equivalence test anchors to.
* The production-method combination uses ``METHOD_TEMPERATURE * logsumexp(c /
  METHOD_TEMPERATURE)``; ``logsumexp`` handles the ``-inf`` entries for methods a
  sequence cannot have been produced by, and the always-finite repetition
  component keeps every column from being all ``-inf``.

Everything else — the unique-sequence table, per-trial gather indices, the
same-length restriction enforced in ``prepare_observed``, and the per-trial
``p_left`` / observed Bernoulli — is identical to the other two adapters.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_SEQ_LEN = 8
N_STATES = 6
_EMITS = "HTHTHT"

# Smoothing temperatures (analyst-set hyperparameters, not fit parameters).
# tau -> 0 recovers motif_stack (Viterbi); tau = 1 recovers motif_stack_softmax.
PATH_TEMPERATURE = 0.35
METHOD_TEMPERATURE = 0.35


# --- Data preparation (numpy; runs once per fit, not inside the graph) ------


def _clean_sequence(seq: str) -> str:
    """Uppercase an H/T sequence and reject anything else."""
    out = "".join(c.upper() for c in str(seq).strip() if not c.isspace())
    bad = sorted({c for c in out if c not in {"H", "T"}})
    if bad:
        raise ValueError(f"Sequence contains non-H/T symbols: {bad}")
    if not out:
        raise ValueError("Sequence must not be empty")
    if len(out) > MAX_SEQ_LEN:
        raise ValueError(
            f"sequence {out!r} is longer than the supported maximum of "
            f"{MAX_SEQ_LEN} symbols"
        )
    return out


def _memory_flags(seq: str) -> tuple[int, int, int]:
    """``(mirror, complement, duplication)`` production-method indicators."""
    n = len(seq)
    prefix_length = (n + 1) // 2
    prefix = seq[:prefix_length]
    mirrored_source = prefix[:-1] if n % 2 else prefix
    suffix = seq[prefix_length:]
    complement = {"H": "T", "T": "H"}
    return (
        int(suffix == mirrored_source[::-1]),
        int(suffix == "".join(complement[symbol] for symbol in mirrored_source[::-1])),
        int(n % 2 == 0 and suffix == prefix),
    )


def _emission_masks(sequences: list[str]) -> np.ndarray:
    """Per-position state-compatibility masks, shape ``(MAX_SEQ_LEN, U, N_STATES)``."""
    masks = np.zeros((MAX_SEQ_LEN, len(sequences), N_STATES), dtype="float64")
    for position in range(MAX_SEQ_LEN):
        for u, seq in enumerate(sequences):
            symbol = "H" if position < len(seq) and seq[position] == "H" else "T"
            for state in range(N_STATES):
                masks[position, u, state] = float(_EMITS[state] == symbol)
    return masks


def unique_sequences(rows) -> list[str]:
    """The distinct cleaned sequences appearing in ``rows``, in a stable order."""
    seen: dict[str, None] = {}
    for row in rows:
        for key in ("sequence_a", "sequence_b"):
            seen.setdefault(_clean_sequence(row[key]), None)
    return sorted(seen)


def _unique_sequence_table(sequences: list[str]) -> dict:
    flags = [_memory_flags(seq) for seq in sequences]
    return {
        "seq_len": np.array([len(seq) for seq in sequences], dtype="int64"),
        "emission_mask": _emission_masks(sequences),
        "mirror_symmetry": np.array([f[0] for f in flags], dtype="int64"),
        "complement_symmetry": np.array([f[1] for f in flags], dtype="int64"),
        "duplication": np.array([f[2] for f in flags], dtype="int64"),
    }


def prepare_observed(rows) -> dict:
    """Build every ``pm.Data`` array for this model from raw stimulus rows."""
    rows = list(rows)
    if not rows:
        raise ValueError("prepare_observed requires at least one row.")

    sequences_a: list[str] = []
    sequences_b: list[str] = []
    for i, row in enumerate(rows):
        missing = [key for key in ("sequence_a", "sequence_b") if key not in row]
        if missing:
            raise ValueError(
                f"motif_stack_tempered needs the raw H/T sequence columns {missing} "
                f"to build its unique-sequence table; row {i} has {sorted(row)}."
            )
        seq_a = _clean_sequence(row["sequence_a"])
        seq_b = _clean_sequence(row["sequence_b"])
        if len(seq_a) != len(seq_b):
            raise ValueError(
                "motif_stack_tempered requires same-length alternatives on every "
                f"trial: row {i} pairs a length-{len(seq_a)} sequence with a "
                f"length-{len(seq_b)} one ({seq_a!r} vs {seq_b!r})."
            )
        sequences_a.append(seq_a)
        sequences_b.append(seq_b)

    sequences = unique_sequences(rows)
    index = {seq: i for i, seq in enumerate(sequences)}

    has_response = ["chose_left" in row for row in rows]
    if any(has_response) and not all(has_response):
        raise ValueError(
            "chose_left is present on some rows but not others "
            f"({sum(has_response)}/{len(rows)}); pass real responses for every "
            "row or none at all."
        )
    if all(has_response):
        chose_left = np.array(
            [int(float(row["chose_left"])) for row in rows], dtype="int64"
        )
    else:
        chose_left = np.zeros(len(rows), dtype="int64")

    return {
        **_unique_sequence_table(sequences),
        "idx_a": np.array([index[seq] for seq in sequences_a], dtype="int64"),
        "idx_b": np.array([index[seq] for seq in sequences_b], dtype="int64"),
        "chose_left": chose_left,
    }


# --- The graph -------------------------------------------------------------


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


def _tempered_forward_log_probabilities(seq_len, emission_mask, init, transition):
    """Tempered log regularity score of the full sequence and its first half.

    Runs the forward recursion with the factors raised to ``1/PATH_TEMPERATURE``
    and scales each prefix score by ``PATH_TEMPERATURE`` — i.e. ``tau * log Z``
    of the ``1/tau``-tempered model. Exact forward at ``tau = 1``; approaches the
    Viterbi max as ``tau -> 0``.
    """
    inverse_temperature = 1.0 / PATH_TEMPERATURE
    init_t = init**inverse_temperature
    transition_t = transition**inverse_temperature

    forward = init_t[None, :] * emission_mask[0]
    prefix_logs = [PATH_TEMPERATURE * pt.log(pt.sum(forward, axis=1))]
    for position in range(1, MAX_SEQ_LEN):
        path_probabilities = forward[:, :, None] * transition_t[None, :, :]
        forward = pt.sum(path_probabilities, axis=1) * emission_mask[position]
        prefix_logs.append(PATH_TEMPERATURE * pt.log(pt.sum(forward, axis=1)))

    by_length = pt.stack(prefix_logs, axis=1)
    full_index = pt.cast(seq_len - 1, "int64")[:, None]
    half_index = pt.cast((seq_len - 1) // 2, "int64")[:, None]
    full = pt.take_along_axis(by_length, full_index, axis=1)[:, 0]
    half = pt.take_along_axis(by_length, half_index, axis=1)[:, 0]
    return full, half


def _log_p_regular(
    seq_len,
    emission_mask,
    mirror_symmetry,
    complement_symmetry,
    duplication,
    init,
    transition,
    method_weights,
):
    full_log_probability, half_log_probability = _tempered_forward_log_probabilities(
        seq_len, emission_mask, init, transition
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
    # Temperature softmax over the production methods: tau * logsumexp(c / tau).
    return METHOD_TEMPERATURE * pt.logsumexp(component_logs / METHOD_TEMPERATURE, axis=0)


# A one-sequence, one-trial placeholder built through the real preparation path.
_PLACEHOLDER = prepare_observed([{"sequence_a": "HT", "sequence_b": "HT"}])


with pm.Model() as model:
    seq_len = pm.Data("seq_len", _PLACEHOLDER["seq_len"])
    emission_mask = pm.Data("emission_mask", _PLACEHOLDER["emission_mask"])
    mirror_symmetry = pm.Data("mirror_symmetry", _PLACEHOLDER["mirror_symmetry"])
    complement_symmetry = pm.Data(
        "complement_symmetry", _PLACEHOLDER["complement_symmetry"]
    )
    duplication = pm.Data("duplication", _PLACEHOLDER["duplication"])
    idx_a = pm.Data("idx_a", _PLACEHOLDER["idx_a"])
    idx_b = pm.Data("idx_b", _PLACEHOLDER["idx_b"])
    chose_left = pm.Data("chose_left", _PLACEHOLDER["chose_left"])

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
        [repetition_weight, mirror_weight, complement_weight, duplication_weight]
    )

    log_regular = _log_p_regular(
        seq_len,
        emission_mask,
        mirror_symmetry,
        complement_symmetry,
        duplication,
        init,
        transition,
        method_weights,
    )
    score = pt.cast(seq_len, "float64") * np.log(0.5) - log_regular

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score[idx_a] - score[idx_b]) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
