"""Load a stimulus list (JSON pairs of H/T sequences) for the recovery harnesses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping


def load_stimuli(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Stimuli file must contain a list: {path}")
    stimuli = []
    for item in data:
        if (
            not isinstance(item, Mapping)
            or "sequence_a" not in item
            or "sequence_b" not in item
        ):
            raise ValueError(f"Invalid stimulus item: {item!r}")
        stimuli.append(
            {
                "sequence_a": str(item["sequence_a"]),
                "sequence_b": str(item["sequence_b"]),
            }
        )
    return stimuli
