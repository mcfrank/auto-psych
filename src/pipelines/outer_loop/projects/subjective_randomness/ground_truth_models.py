"""
Ground-truth models for subjective_randomness. Used only for --ground-truth-model
(data generation to verify the loop recovers the generative process). Not used
for hypothesis generation or for assessing the theorist's models.

Each model: (stimulus, response_options) -> dict[response, probability].
Stimulus is (seq_a, seq_b); response_options e.g. ["left", "right"].
"""

import math
from typing import Dict, List, Tuple

Stimulus = Tuple[str, str]


def _n_heads(seq: str) -> int:
    return sum(1 for c in seq if c.upper() == "H")


def _n_alternations(seq: str) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _imbalance(seq: str) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    return 2.0 * abs((_n_heads(seq) / n) - 0.5)


def _alternation_rate(seq: str) -> float:
    return _n_alternations(seq) / (len(seq) - 1) if len(seq) > 1 else 0.0


def _late_alternation_rate(seq: str) -> float:
    n = len(seq)
    if n <= 2:
        return 0.0
    start = max(1, n // 2)
    transitions = list(range(1, n))
    late = [i for i in transitions if i >= start]
    if not late:
        return 0.0
    return sum(1 for i in late if seq[i] != seq[i - 1]) / len(late)


def alternation(stimulus: Stimulus, response_options: List[str]) -> Dict[str, float]:
    """
    Prefer sequence with more alternations (H-T or T-H transitions).
    P(choose A) proportional to number of alternations in A.
    """
    seq_a, seq_b = stimulus
    na = _n_alternations(seq_a)
    nb = _n_alternations(seq_b)
    sa = 1.0 + na
    sb = 1.0 + nb
    total = sa + sb
    return {response_options[0]: sa / total, response_options[1]: sb / total}


def prefer_more_heads(stimulus: Stimulus, response_options: List[str]) -> Dict[str, float]:
    """
    Prefer sequence with more heads.
    P(choose A) proportional to number of H in A.
    """
    seq_a, seq_b = stimulus
    ha = _n_heads(seq_a)
    hb = _n_heads(seq_b)
    sa = 1.0 + ha
    sb = 1.0 + hb
    total = sa + sb
    return {response_options[0]: sa / total, response_options[1]: sb / total}


def length_sensitive_alternation(
    stimulus: Stimulus,
    response_options: List[str],
) -> Dict[str, float]:
    """Hidden out-of-family process: alternation/balance with a length bonus.

    The seed families capture alternation, balance, compressibility, and
    diagnosticity, but none has an explicit preference for longer samples as
    more trustworthy evidence. The active feature set exposes length, so the
    theorist can discover a variant with this mechanism.
    """
    seq_a, seq_b = stimulus

    def score(seq: str) -> float:
        n = len(seq)
        length_bonus = math.log(max(n, 1)) / math.log(8)
        return (
            1.20 * _alternation_rate(seq)
            - 0.85 * _imbalance(seq)
            + 0.35 * length_bonus
        )

    p_left = _sigmoid(4.0 * (score(seq_a) - score(seq_b)))
    return {response_options[0]: p_left, response_options[1]: 1.0 - p_left}


def recency_weighted_alternation(
    stimulus: Stimulus,
    response_options: List[str],
) -> Dict[str, float]:
    """Harder hidden process: later transitions matter more than early ones."""
    seq_a, seq_b = stimulus

    def score(seq: str) -> float:
        return 1.35 * _late_alternation_rate(seq) - 0.70 * _imbalance(seq)

    p_left = _sigmoid(4.0 * (score(seq_a) - score(seq_b)))
    return {response_options[0]: p_left, response_options[1]: 1.0 - p_left}


# Only these are used for --ground-truth-model. Do not use for theorist/design/analyze/interpret.
GROUND_TRUTH_MODELS: Dict[str, callable] = {
    "alternation": alternation,
    "prefer_more_heads": prefer_more_heads,
    "length_sensitive_alternation": length_sensitive_alternation,
    "recency_weighted_alternation": recency_weighted_alternation,
}
