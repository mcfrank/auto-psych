"""Shared types and prediction API for subjective-randomness-style models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Type for stimulus: (sequence_a, sequence_b), each sequence is string of H/T
Stimulus = Tuple[str, str]
# For APIs that accept dict from JSON: {"sequence_a": str, "sequence_b": str}
StimulusLike = Union[Stimulus, Dict[str, str]]


def _normalize_stimulus(stimulus: Stimulus | dict) -> Stimulus:
    """Accept (seq_a, seq_b) or dict with sequence_a, sequence_b; return (seq_a, seq_b)."""
    if isinstance(stimulus, (list, tuple)) and len(stimulus) >= 2:
        return (str(stimulus[0]), str(stimulus[1]))
    if (
        isinstance(stimulus, dict)
        and "sequence_a" in stimulus
        and "sequence_b" in stimulus
    ):
        return (str(stimulus["sequence_a"]), str(stimulus["sequence_b"]))
    raise ValueError(
        f"Stimulus must be (seq_a, seq_b) or dict with sequence_a, sequence_b; got {type(stimulus)}"
    )


def get_model_predictions(
    stimulus: StimulusLike,
    response_options: List[str],
    model_names: List[str],
    theorist_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Return predictions for each model: { model_name: { response: prob } }.
    Models are resolved only from theorist_dir (1_theory/<name>.py). No global library.
    stimulus may be a tuple (seq_a, seq_b) or a dict with keys sequence_a, sequence_b.

    Every requested model appears in the result, or the call raises. A model
    that cannot be loaded or that crashes while predicting is NOT dropped:
    downstream consumers (EIG, ground-truth generation, model weighting)
    renormalize over whatever they receive, so a silently missing model changes
    the science while looking like a smaller hypothesis space. If you genuinely
    want to skip unusable models, screen them explicitly first — see
    ``src.pipelines.outer_loop.eig._screen_usable_models``.
    """
    from src.models.theorist.loader import get_model_callable

    stimulus = _normalize_stimulus(stimulus)
    out = {}
    for name in model_names:
        fn = get_model_callable(name, theorist_dir)
        out[name] = fn(stimulus, response_options)
    return out
