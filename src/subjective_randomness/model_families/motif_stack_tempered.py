"""Temperature-controlled smoothing of the four-motif stack automaton.

This is a single family that interpolates between the Viterbi ``motif_stack``
(hard max) and the fully-marginalising ``motif_stack_softmax`` (temperature 1),
via a temperature ``tau`` on each of the two maxes:

* ``PATH_TEMPERATURE`` softens the max over hidden HMM paths (Viterbi at
  ``tau -> 0``, forward/exact marginal at ``tau = 1``);
* ``METHOD_TEMPERATURE`` softens the max over the four production methods (argmax
  at ``tau -> 0``, mixture at ``tau = 1``).

The smoothing operator is the temperature softmax ``tau * logsumexp(v / tau)``.
As ``tau -> 0`` it returns ``max(v)`` exactly, recovering the original
predictions; as ``tau`` grows it rounds off the ridge where the argmax switches.
Any ``tau > 0`` is C-infinity smooth, so it removes the non-differentiable kinks
that make the Viterbi likelihood awkward for NUTS — but the curvature near the
old ridge scales like ``1/tau``, so a small ``tau`` keeps predictions close to
the hard max at the cost of a stiffer geometry.  The point of this family is to
pick a ``tau`` on that trade-off, rather than being forced to the ``tau = 0``
(hard, hard-to-fit) or ``tau = 1`` (smooth, prediction-shifting) extremes.

Endpoints (pinned by ``tests/test_motif_stack_tempered.py``):

* ``(path=0, method=0)`` reproduces ``motif_stack`` to floating-point noise;
* ``(path=1, method=1)`` reproduces ``motif_stack_softmax``.

``PATH_TEMPERATURE`` / ``METHOD_TEMPERATURE`` are model hyperparameters set by
the analyst, NOT fitted parameters: the free parameters and their bounds are
exactly those of ``motif_stack``.  The defaults are the smoothing recommended by
``scripts/subjective_randomness/analyze_motif_stack_smoothing_tradeoff.py`` — a
modest softening that keeps predictions close to the Viterbi model.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

from .common import (
    Stimulus,
    choice_probability,
    clean_sequence,
    clipped,
    distribution,
    merge_params,
    normalize_stimulus,
)

MODEL_NAME = "motif_stack_tempered"
N_STATES = 6
_EMITS = "HTHTHT"
_NEG_INF = float("-inf")

# Default smoothing temperatures (analyst-set hyperparameters, not fit params).
# tau -> 0 recovers the Viterbi motif_stack; tau = 1 the full marginal
# motif_stack_softmax. See the module docstring / tradeoff script.
DEFAULT_PATH_TEMPERATURE = 0.35
DEFAULT_METHOD_TEMPERATURE = 0.35

DEFAULT_PARAMS: Dict[str, float] = {
    "delta": 0.5493,
    "alpha": 0.2073,
    "repetition_weight": 0.6839,
    "mirror_share": 0.4989,
    "complement_share": 0.3984,
    "beta": 1.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "delta": (0.01, 0.99),
    "alpha": (0.01, 0.99),
    "repetition_weight": (0.001, 0.999),
    "mirror_share": (0.001, 0.999),
    "complement_share": (0.001, 0.999),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def _logsumexp(values: Sequence[float]) -> float:
    m = max(values)
    if m == _NEG_INF:
        return _NEG_INF
    return m + math.log(math.fsum(math.exp(v - m) for v in values))


def _softmax_combine(values: Sequence[float], temperature: float) -> float:
    """``temperature * logsumexp(values / temperature)``; ``max(values)`` at temp<=0.

    The single smoothing operator used for both maxes. It equals ``max`` exactly
    in the ``temperature <= 0`` limit and the log-sum (mixture) at
    ``temperature == 1``.
    """
    if temperature <= 0.0:
        return max(values)
    return temperature * _logsumexp([v / temperature for v in values])


def _matrices(delta: float, alpha: float) -> tuple[List[float], List[List[float]]]:
    a, a2, d = clipped(alpha), clipped(alpha) ** 2, clipped(delta)
    rows = [
        [d, a, a2, 0.0, 0.0, a2],
        [a, d, a2, 0.0, 0.0, a2],
        [a, a, 0.0, d, 0.0, a2],
        [a, a, d, 0.0, 0.0, a2],
        [a, a, a2, 0.0, 0.0, d],
        [a, a, a2, 0.0, d, 0.0],
    ]
    transition = [[value / sum(row) for value in row] for row in rows]
    init_raw = [a, a, a2, 0.0, 0.0, a2]
    init_total = sum(init_raw)
    return [value / init_total for value in init_raw], transition


def _tempered_prefix_logscores(
    seq: str, delta: float, alpha: float, path_temperature: float
) -> List[float]:
    """Tempered log regularity score of each observed prefix.

    Runs the hidden-path recursion in log space, combining the previous state
    with ``max`` (``path_temperature <= 0``, Viterbi) or ``logsumexp`` on the
    ``1/tau``-scaled log-factors (``path_temperature > 0``).  The prefix score is
    ``tau * logsumexp`` over states — i.e. ``tau * log Z`` of the model whose
    factors are raised to ``1/tau`` — which is ``max``-path log-probability as
    ``tau -> 0`` and the exact forward log-probability at ``tau = 1``.
    """
    init, transition = _matrices(delta, alpha)
    log_init = [math.log(v) if v > 0.0 else _NEG_INF for v in init]
    log_transition = [
        [math.log(v) if v > 0.0 else _NEG_INF for v in row] for row in transition
    ]
    viterbi = path_temperature <= 0.0
    scale = 1.0 if viterbi else 1.0 / path_temperature

    def combine(values: Sequence[float]) -> float:
        return max(values) if viterbi else _logsumexp(values)

    def reduce_states(state_logs: Sequence[float]) -> float:
        return max(state_logs) if viterbi else path_temperature * _logsumexp(state_logs)

    state_logs = [
        scale * log_init[state] if _EMITS[state] == seq[0] else _NEG_INF
        for state in range(N_STATES)
    ]
    scores = [reduce_states(state_logs)]
    for symbol in seq[1:]:
        next_logs = []
        for state in range(N_STATES):
            if _EMITS[state] != symbol:
                next_logs.append(_NEG_INF)
                continue
            transitions = [
                state_logs[previous] + scale * log_transition[previous][state]
                for previous in range(N_STATES)
            ]
            next_logs.append(combine(transitions))
        state_logs = next_logs
        scores.append(reduce_states(state_logs))
    return scores


def _method_weights(params: Mapping[str, float]) -> Dict[str, float]:
    repetition = clipped(float(params["repetition_weight"]))
    mirror_share = clipped(float(params["mirror_share"]))
    complement_share = clipped(float(params["complement_share"]))
    remaining = 1.0 - repetition
    mirror = remaining * mirror_share
    remaining_after_mirror = remaining * (1.0 - mirror_share)
    complement = remaining_after_mirror * complement_share
    duplication = remaining_after_mirror * (1.0 - complement_share)
    return {
        "repetition": repetition,
        "mirror": mirror,
        "complement": complement,
        "duplication": duplication,
    }


def _memory_patterns(seq: str) -> Dict[str, bool]:
    prefix_length = (len(seq) + 1) // 2
    prefix = seq[:prefix_length]
    mirrored_source = prefix[:-1] if len(seq) % 2 else prefix
    suffix = seq[prefix_length:]
    complement = {"H": "T", "T": "H"}
    return {
        "mirror": suffix == mirrored_source[::-1],
        "complement": suffix
        == "".join(complement[symbol] for symbol in mirrored_source[::-1]),
        "duplication": len(seq) % 2 == 0 and suffix == prefix,
    }


def log_p_regular(
    seq: str,
    params: Mapping[str, float] | None = None,
    *,
    path_temperature: float = DEFAULT_PATH_TEMPERATURE,
    method_temperature: float = DEFAULT_METHOD_TEMPERATURE,
) -> float:
    p = merge_params(DEFAULT_PARAMS, params)
    s = clean_sequence(seq)
    scores = _tempered_prefix_logscores(
        s, float(p["delta"]), float(p["alpha"]), path_temperature
    )
    weights = _method_weights(p)
    full_log = scores[-1]
    half_log = scores[(len(s) + 1) // 2 - 1]
    components = [math.log(weights["repetition"]) + full_log]
    for method, matches in _memory_patterns(s).items():
        if matches:
            components.append(math.log(weights[method]) + half_log)
    regular_log = _softmax_combine(components, method_temperature)
    if not math.isfinite(regular_log):
        raise ValueError(f"Motif stack assigned zero probability to {s!r}")
    return regular_log


def score_sequence(
    seq: str,
    params: Mapping[str, float] | None = None,
    *,
    path_temperature: float = DEFAULT_PATH_TEMPERATURE,
    method_temperature: float = DEFAULT_METHOD_TEMPERATURE,
) -> float:
    s = clean_sequence(seq)
    return len(s) * math.log(0.5) - log_p_regular(
        s, params, path_temperature=path_temperature, method_temperature=method_temperature
    )


def predict_left(
    stimulus: Stimulus | Mapping[str, str],
    params: Mapping[str, float] | None = None,
    *,
    path_temperature: float = DEFAULT_PATH_TEMPERATURE,
    method_temperature: float = DEFAULT_METHOD_TEMPERATURE,
) -> float:
    seq_a, seq_b = normalize_stimulus(stimulus)
    if len(seq_a) != len(seq_b):
        raise ValueError(
            "motif_stack_tempered is defined only for same-length sequence "
            f"comparisons; got lengths {len(seq_a)} and {len(seq_b)}"
        )
    p = merge_params(DEFAULT_PARAMS, params)
    score_a = score_sequence(
        seq_a, p, path_temperature=path_temperature, method_temperature=method_temperature
    )
    score_b = score_sequence(
        seq_b, p, path_temperature=path_temperature, method_temperature=method_temperature
    )
    return choice_probability(score_a, score_b, p)


def predict(
    stimulus: Stimulus | Mapping[str, str],
    response_options: Sequence[str] = ("left", "right"),
    params: Mapping[str, float] | None = None,
    *,
    path_temperature: float = DEFAULT_PATH_TEMPERATURE,
    method_temperature: float = DEFAULT_METHOD_TEMPERATURE,
) -> Dict[str, float]:
    return distribution(
        predict_left(
            stimulus,
            params,
            path_temperature=path_temperature,
            method_temperature=method_temperature,
        ),
        response_options,
    )
