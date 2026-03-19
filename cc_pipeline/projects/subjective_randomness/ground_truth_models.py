"""
Ground-truth models for subjective_randomness. Used only for --ground-truth-model
(data generation to verify the loop recovers the generative process). Not used
for hypothesis generation or for assessing the theorist's models.

Each model: (stimulus, response_options) -> dict[response, probability].
Stimulus is (seq_a, seq_b); response_options e.g. ["left", "right"].
"""

from typing import Dict, List, Tuple

Stimulus = Tuple[str, str]


def _n_heads(seq: str) -> int:
    return sum(1 for c in seq if c.upper() == "H")


def _n_alternations(seq: str) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])


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


# Only these are used for --ground-truth-model. Do not use for theorist/design/analyze/interpret.
GROUND_TRUTH_MODELS: Dict[str, callable] = {
    "alternation": alternation,
    "prefer_more_heads": prefer_more_heads,
}
