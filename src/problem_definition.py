"""
Parse structured fields from problem_definition.md for use by designer and other agents.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from src.config import problem_definition_path

DEFAULT_TOTAL_TRIALS = 30
DEFAULT_ALLOWED_SEQUENCE_LENGTHS = [4, 6, 8]


def parse_problem_definition(project_id: str) -> Dict[str, Any]:
    """
    Read problem_definition.md and return a dict with optional:
    - total_trials: int (default 30)
    - allowed_sequence_lengths: list of int (default [4, 6, 8])
    """
    path = problem_definition_path(project_id)
    if not path.exists():
        return {
            "total_trials": DEFAULT_TOTAL_TRIALS,
            "allowed_sequence_lengths": DEFAULT_ALLOWED_SEQUENCE_LENGTHS,
        }
    text = path.read_text(encoding="utf-8")

    total_trials = DEFAULT_TOTAL_TRIALS
    # e.g. "Total trials per experiment: 30" or "total_trials: 30"
    m = re.search(r"(?:Total trials per experiment|total_trials)\s*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        total_trials = int(m.group(1))

    allowed_lengths = DEFAULT_ALLOWED_SEQUENCE_LENGTHS
    # e.g. "Allowed sequence lengths: 4, 6, 8" or "Sequence lengths: 4, 6, 8"
    m = re.search(r"(?:Allowed )?sequence lengths?\s*:\s*([\d,\s\-]+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1)
        lengths = []
        for part in re.split(r"[\s,]+", raw):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                try:
                    lo, hi = int(a.strip()), int(b.strip())
                    lengths.extend(range(lo, hi + 1))
                except ValueError:
                    pass
            elif part.isdigit():
                lengths.append(int(part))
        if lengths:
            allowed_lengths = sorted(set(lengths))

    return {
        "total_trials": total_trials,
        "allowed_sequence_lengths": allowed_lengths,
    }
