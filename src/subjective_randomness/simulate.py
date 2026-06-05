"""Simulate subjective-randomness choices from a parametric model family."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

RESPONSE_COLUMNS = [
    "participant_id",
    "trial_index",
    "sequence_a",
    "sequence_b",
    "chose_left",
    "chose_right",
    "model",
    "true_params",
]


def load_stimuli(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Stimuli file must contain a list: {path}")
    stimuli = []
    for item in data:
        if not isinstance(item, Mapping) or "sequence_a" not in item or "sequence_b" not in item:
            raise ValueError(f"Invalid stimulus item: {item!r}")
        stimuli.append({"sequence_a": str(item["sequence_a"]), "sequence_b": str(item["sequence_b"])})
    return stimuli


def generate_rows(
    model_module,
    stimuli: Iterable[Mapping[str, str]],
    params: Mapping[str, float],
    n_participants: int,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    model_name = getattr(model_module, "MODEL_NAME", model_module.__name__.split(".")[-1])
    params_json = json.dumps(dict(params), sort_keys=True)
    stimuli_list = list(stimuli)
    for participant_id in range(n_participants):
        for trial_index, stim in enumerate(stimuli_list):
            p_left = float(model_module.predict_left(stim, params))
            chose_left = 1 if rng.random() < p_left else 0
            rows.append(
                {
                    "participant_id": participant_id,
                    "trial_index": trial_index,
                    "sequence_a": stim["sequence_a"],
                    "sequence_b": stim["sequence_b"],
                    "chose_left": chose_left,
                    "chose_right": 1 - chose_left,
                    "model": model_name,
                    "true_params": params_json,
                }
            )
    return rows


def write_rows(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESPONSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
