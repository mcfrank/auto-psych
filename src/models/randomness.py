"""
Subjective randomness: probabilistic models for which sequence looks "more random".

Each model is a function (stimulus, response_options) -> dict[response, probability].
Stimulus is a pair of sequences (seq_a, seq_b); response_options e.g. ["left", "right"].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Type for stimulus: (sequence_a, sequence_b), each sequence is string of H/T
Stimulus = Tuple[str, str]
# For APIs that accept dict from JSON: {"sequence_a": str, "sequence_b": str}
StimulusLike = Union[Stimulus, Dict[str, str]]


def _likelihood_fair(seq: str) -> float:
    """P(sequence | fair coin) = 0.5^len."""
    return 0.5 ** len(seq)


def _n_heads(seq: str) -> int:
    return sum(1 for c in seq if c.upper() == "H")


def _n_alternations(seq: str) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])


def _bernoulli_likelihood(seq: str, p: float) -> float:
    """Likelihood of sequence under Bernoulli(p)."""
    if p <= 0 or p >= 1:
        return 1e-10
    n_h = _n_heads(seq)
    n_t = len(seq) - n_h
    return (p ** n_h) * ((1 - p) ** n_t)


def bayesian_fair_coin(stimulus: Stimulus, response_options: List[str]) -> Dict[str, float]:
    """
    Choose with probability proportional to likelihood under fair coin.
    P(choose A) = L(A) / (L(A) + L(B)); under fair coin L(A)=L(B)=0.5^n so 50-50.
    """
    seq_a, seq_b = stimulus
    la = _likelihood_fair(seq_a)
    lb = _likelihood_fair(seq_b)
    total = la + lb
    if total <= 0:
        total = 1.0
    p_a = la / total
    p_b = lb / total
    return {response_options[0]: p_a, response_options[1]: p_b}


def representativeness(stimulus: Stimulus, response_options: List[str]) -> Dict[str, float]:
    """
    Prefer sequence closer to 50/50 balance. Score = 1 / (1 + |n_H - n_T|).
    P(choose A) proportional to score(A).
    """
    seq_a, seq_b = stimulus
    def score(s: str) -> float:
        n_h = _n_heads(s)
        n_t = len(s) - n_h
        imbalance = abs(n_h - n_t)
        return 1.0 / (1.0 + imbalance)
    sa = score(seq_a)
    sb = score(seq_b)
    total = sa + sb
    if total <= 0:
        total = 1.0
    return {response_options[0]: sa / total, response_options[1]: sb / total}


def alternation(stimulus: Stimulus, response_options: List[str]) -> Dict[str, float]:
    """
    Prefer sequence with more alternations (H-T or T-H transitions).
    P(choose A) proportional to number of alternations in A.
    """
    seq_a, seq_b = stimulus
    na = _n_alternations(seq_a)
    nb = _n_alternations(seq_b)
    # Avoid zero total: use 1 + n_alt so both have positive weight
    sa = 1.0 + na
    sb = 1.0 + nb
    total = sa + sb
    return {response_options[0]: sa / total, response_options[1]: sb / total}


def subjective_generator(stimulus: Stimulus, response_options: List[str]) -> Dict[str, float]:
    """
    Infer bias p from the two sequences (e.g. pooled proportion of heads),
    then P(choose A) = likelihood(A | p) / (likelihood(A|p) + likelihood(B|p)).
    """
    seq_a, seq_b = stimulus
    n_h_a = _n_heads(seq_a)
    n_h_b = _n_heads(seq_b)
    n = len(seq_a) + len(seq_b)
    p_inferred = (n_h_a + n_h_b) / n if n else 0.5
    p_inferred = max(0.01, min(0.99, p_inferred))
    la = _bernoulli_likelihood(seq_a, p_inferred)
    lb = _bernoulli_likelihood(seq_b, p_inferred)
    total = la + lb
    if total <= 0:
        total = 1.0
    return {response_options[0]: la / total, response_options[1]: lb / total}


# Registry for the theorist: name -> function
MODEL_LIBRARY: Dict[str, callable] = {
    "bayesian_fair_coin": bayesian_fair_coin,
    "representativeness": representativeness,
    "alternation": alternation,
    "subjective_generator": subjective_generator,
}


def _normalize_stimulus(stimulus: Stimulus | dict) -> Stimulus:
    """Accept (seq_a, seq_b) or dict with sequence_a, sequence_b; return (seq_a, seq_b)."""
    if isinstance(stimulus, (list, tuple)) and len(stimulus) >= 2:
        return (str(stimulus[0]), str(stimulus[1]))
    if isinstance(stimulus, dict) and "sequence_a" in stimulus and "sequence_b" in stimulus:
        return (str(stimulus["sequence_a"]), str(stimulus["sequence_b"]))
    raise ValueError(f"Stimulus must be (seq_a, seq_b) or dict with sequence_a, sequence_b; got {type(stimulus)}")


def get_model_predictions(
    stimulus: StimulusLike,
    response_options: List[str],
    model_names: List[str],
    theorist_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Return predictions for each model: { model_name: { response: prob } }.
    If theorist_dir is set, models are resolved from run-dir <model_name>.py first,
    then MODEL_LIBRARY.
    stimulus may be a tuple (seq_a, seq_b) or a dict with keys sequence_a, sequence_b.
    """
    from src.models.loader import get_model_callable

    stimulus = _normalize_stimulus(stimulus)
    out = {}
    for name in model_names:
        try:
            fn = get_model_callable(name, theorist_dir)
            out[name] = fn(stimulus, response_options)
        except (KeyError, FileNotFoundError, ValueError):
            continue
    return out
