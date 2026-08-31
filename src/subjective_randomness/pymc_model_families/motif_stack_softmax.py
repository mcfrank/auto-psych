"""PyMC adapter for the softmax (marginalising) four-motif stack automaton.

This is :mod:`.motif_stack` with both of its maxes replaced by their
temperature-1 softmax (log-sum-exp) counterparts, so ``log P(x | regular)`` is
the exact marginal likelihood rather than the max over the single best hidden
path and production method:

* the Viterbi max-product recursion becomes the forward sum-product recursion —
  ``pt.sum`` over the previous state instead of ``pt.max``;
* the ``pt.max`` over the four production methods becomes ``pt.logsumexp`` over
  them, i.e. the mixture ``sum_M P(M) P(x | M)`` in the log domain.

Everything else — the unique-sequence table, the per-trial gather indices, the
same-length restriction enforced in Python by ``prepare_observed``, and the
``p_left`` / observed-Bernoulli per-trial structure — is identical to the
Viterbi adapter, so this file agrees with its own pure-Python twin
(``model_families.motif_stack_softmax``) to floating-point noise.

Why it exists
-------------
The Viterbi likelihood has non-differentiable ridges wherever the argmax (over
paths or over methods) switches; that geometry is why the Viterbi adapter lowers
``target_accept`` and still takes long NUTS trajectories.  The marginal
likelihood is smooth in the parameters everywhere, so this variant is a
candidate for being materially cheaper to sample.  It therefore does NOT declare
its own ``SAMPLER_SETTINGS``: it inherits the production ``target_accept`` rather
than the Viterbi model's lowered-``0.9`` workaround, which the smoothing should
make unnecessary (verified empirically by
``scripts/subjective_randomness/compare_motif_stack_fit_difficulty.py``).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_SEQ_LEN = 8
N_STATES = 6
_EVEN_STATES = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
# State s emits "H" when _EVEN_STATES[s] == 1 (i.e. "HTHTHT"[s] == "H").
_EMITS = "HTHTHT"


# --- Data preparation (numpy; runs once per fit, not inside the graph) ------


def _clean_sequence(seq: str) -> str:
    """Uppercase an H/T sequence and reject anything else.

    Kept local rather than imported from ``src.subjective_randomness.features``:
    this file is also staged into a coding agent's working directory as a seed
    model, where the ``src`` package is not importable.
    """
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
    """``(mirror, complement, duplication)`` production-method indicators.

    Each memory method generates the first half with the motif HMM and affixes,
    respectively, its reversal, complemented reversal, or duplicate. For odd
    lengths the middle symbol belongs to the generated prefix and is not
    repeated.
    """
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
    """Per-position state-compatibility masks, shape ``(MAX_SEQ_LEN, U, N_STATES)``.

    ``mask[t, u, s]`` is 1.0 when state ``s`` can emit sequence ``u``'s symbol at
    position ``t``. Positions past a sequence's length are padded with symbol
    "T" (0); those prefix entries are computed but never selected, because
    ``seq_len`` indexes only positions within the sequence.
    """
    masks = np.zeros((MAX_SEQ_LEN, len(sequences), N_STATES), dtype="float64")
    for position in range(MAX_SEQ_LEN):
        for u, seq in enumerate(sequences):
            symbol = "H" if position < len(seq) and seq[position] == "H" else "T"
            for state in range(N_STATES):
                masks[position, u, state] = float(_EMITS[state] == symbol)
    return masks


def unique_sequences(rows) -> list[str]:
    """The distinct cleaned sequences appearing in ``rows``, in a stable order.

    Exposed so callers and tests can relate ``idx_a`` / ``idx_b`` back to the
    sequences they index.
    """
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
    """Build every ``pm.Data`` array for this model from raw stimulus rows.

    ``rows`` are dicts carrying raw ``sequence_a`` / ``sequence_b`` H/T strings,
    and ``chose_left`` when real responses exist. Prediction paths (EIG design
    pools, ``predict_p_left_draws``) pass stimuli with no responses; those get
    dummy zeros, since ``p_left`` does not depend on the observed value.

    Recognised by ``src.models.pymc_inference`` in place of its default
    one-CSV-column-per-container mapping, which cannot express a table indexed
    by gather indices.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("prepare_observed requires at least one row.")

    sequences_a: list[str] = []
    sequences_b: list[str] = []
    for i, row in enumerate(rows):
        missing = [key for key in ("sequence_a", "sequence_b") if key not in row]
        if missing:
            raise ValueError(
                f"motif_stack_softmax needs the raw H/T sequence columns {missing} "
                f"to build its unique-sequence table; row {i} has {sorted(row)}."
            )
        seq_a = _clean_sequence(row["sequence_a"])
        seq_b = _clean_sequence(row["sequence_b"])
        if len(seq_a) != len(seq_b):
            raise ValueError(
                "motif_stack_softmax requires same-length alternatives on every "
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


def _forward_log_probabilities(seq_len, emission_mask, init, transition):
    """Log forward probabilities for the full sequence and its first half.

    The sum-product counterpart of the Viterbi adapter's recursion: ``pt.sum``
    over the previous state where the Viterbi version takes ``pt.max``, so each
    entry is the marginal probability of the observed prefix rather than the best
    path's joint probability. Operates on the unique-sequence batch:
    ``emission_mask`` is ``(MAX_SEQ_LEN, U, N_STATES)`` and the returned tensors
    are ``(U,)``.
    """
    forward = init[None, :] * emission_mask[0]
    prefix_logs = [pt.log(pt.sum(forward, axis=1))]
    for position in range(1, MAX_SEQ_LEN):
        path_probabilities = forward[:, :, None] * transition[None, :, :]
        forward = pt.sum(path_probabilities, axis=1) * emission_mask[position]
        prefix_logs.append(pt.log(pt.sum(forward, axis=1)))

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
    full_log_probability, half_log_probability = _forward_log_probabilities(
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
    # Mixture over production methods (softmax over M) rather than the Viterbi
    # argmax. logsumexp handles the -inf entries for methods x cannot have been
    # produced by; the repetition component is always finite, so no column is
    # all -inf and the gradient stays defined.
    return pt.logsumexp(component_logs, axis=0)


# A one-sequence, one-trial placeholder built through the real preparation path,
# so the containers always start out mutually consistent and valid.
_PLACEHOLDER = prepare_observed([{"sequence_a": "HT", "sequence_b": "HT"}])


with pm.Model() as model:
    # Unique-sequence table, shape (U, ...) — scored once per distinct sequence.
    seq_len = pm.Data("seq_len", _PLACEHOLDER["seq_len"])
    emission_mask = pm.Data("emission_mask", _PLACEHOLDER["emission_mask"])
    mirror_symmetry = pm.Data("mirror_symmetry", _PLACEHOLDER["mirror_symmetry"])
    complement_symmetry = pm.Data(
        "complement_symmetry", _PLACEHOLDER["complement_symmetry"]
    )
    duplication = pm.Data("duplication", _PLACEHOLDER["duplication"])
    # Per-trial columns, shape (T,).
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
        [
            repetition_weight,
            mirror_weight,
            complement_weight,
            duplication_weight,
        ]
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
    # Randomness score per unique sequence: log P(x|fair) - log P(x|regular).
    score = pt.cast(seq_len, "float64") * np.log(0.5) - log_regular

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score[idx_a] - score[idx_b]) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
