"""Result-shaping helpers for deployments.

Mirrors the /submit and /results Cloud Functions (functions/index.js) in Python:
the participant data path runs entirely through those Cloud Functions, which use
their own admin credentials to write/read the Firestore ``responses``
subcollection. The pipeline itself does no server-side Firestore access.
"""

from __future__ import annotations

import csv
import io
from typing import Any


def _chose_left(value: Any) -> bool:
    """Interpret a stored ``chose_left`` (number, bool, or string) as left/right.

    A naive ``bool(value)`` is wrong for strings — ``bool("0")`` and
    ``bool("false")`` are both ``True`` — which would flip a right-choice to left
    when shaping live data. Coerce explicitly.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "left"):
            return True
        if v in ("0", "false", "no", "right", ""):
            return False
        try:
            return float(v) != 0.0
        except ValueError:
            return False
    return bool(value)


def validate_submit_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    trials = payload.get("trials")
    if not isinstance(trials, list):
        return False, "trials must be a list"
    if payload.get("collection_session_id"):
        return True, ""
    if payload.get("project_id") and payload.get("run_id"):
        return True, ""
    return False, "collection_session_id or project_id/run_id is required"


def responses_to_csv(response_docs: list[tuple[str, dict[str, Any]]]) -> str:
    """Flatten Firestore response docs into the pipeline CSV shape."""
    rows: list[dict[str, Any]] = []
    for participant_index, (doc_id, data) in enumerate(response_docs):
        trials = data.get("trials") or []
        for trial_index, trial in enumerate(trials):
            if trial.get("sequence_a") is None or trial.get("sequence_b") is None:
                continue
            chose_left = trial.get("chose_left")
            if chose_left is None:
                continue
            left = _chose_left(chose_left)
            rows.append(
                {
                    "participant_id": participant_index,
                    "participant_id_str": doc_id,
                    "trial_index": trial_index,
                    "sequence_a": str(trial["sequence_a"]),
                    "sequence_b": str(trial["sequence_b"]),
                    "chose_left": 1 if left else 0,
                    "chose_right": 0 if left else 1,
                    "model": "",
                }
            )

    fieldnames = [
        "participant_id",
        "participant_id_str",
        "trial_index",
        "sequence_a",
        "sequence_b",
        "chose_left",
        "chose_right",
        "model",
    ]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()
