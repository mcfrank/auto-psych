from __future__ import annotations

import math


PARAM_NAMES = ["alternation_weight", "balance_weight", "temperature"]
PARAM_BOUNDS = [(-5.0, 5.0), (-5.0, 5.0), (0.1, 5.0)]
INITIAL_PARAMS = [1.0, 1.0, 1.0]


def _alternations(sequence: str) -> int:
    return sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)


def _balance(sequence: str) -> int:
    return -abs(sequence.count("H") - sequence.count("T"))


def cognitive_model(stimulus, response_options, params=None):
    """Parameterized subjective-randomness model.

    Scores each sequence by alternation and H/T balance, then converts the
    score difference into a probability of choosing the left option.
    """
    if params is None:
        params = INITIAL_PARAMS
    alternation_weight, balance_weight, temperature = [
        max(lo, min(hi, float(value)))
        for value, (lo, hi) in zip(params, PARAM_BOUNDS)
    ]
    if isinstance(stimulus, dict):
        sequence_a, sequence_b = stimulus["sequence_a"], stimulus["sequence_b"]
    else:
        sequence_a, sequence_b = stimulus

    def score(sequence: str) -> float:
        return alternation_weight * _alternations(sequence) + balance_weight * _balance(sequence)

    logit = (score(sequence_a) - score(sequence_b)) / max(temperature, 1e-6)
    p_left = 1.0 / (1.0 + math.exp(-logit))
    return {response_options[0]: p_left, response_options[1]: 1.0 - p_left}
