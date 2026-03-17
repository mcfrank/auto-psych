"""
Shared types and prediction API for subjective-randomness-style models.
Models are implemented by the theorist (1_theory/<name>.py) or by the project
(projects/<project>/ground_truth_models.py for --ground-truth-model only).
No global model library in src.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Type for stimulus: (sequence_a, sequence_b), each sequence is string of H/T
Stimulus = Tuple[str, str]
# For APIs that accept dict from JSON: {"sequence_a": str, "sequence_b": str}
StimulusLike = Union[Stimulus, Dict[str, str]]


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
    Models are resolved only from theorist_dir (1_theory/<name>.py). No global library.
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
