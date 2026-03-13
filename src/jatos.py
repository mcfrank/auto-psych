"""JATOS API: fetch results and parse result archive ZIP into pipeline CSV format."""

import io
import json
import re
import zipfile
from typing import Any, Dict, List

import urllib.request


def fetch_jatos_results(
    base_url: str,
    component_id: str,
    token: str,
) -> bytes:
    """
    POST JATOS results export for the given component. Returns the ZIP file bytes.
    Raises on HTTP error.
    """
    base_url = base_url.rstrip("/")
    url = f"{base_url}/jatos/api/v1/results?componentId={component_id}"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/zip",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def parse_jatos_results_zip(zip_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Parse a JATOS results archive ZIP into rows matching our responses.csv schema:
    participant_id, trial_index, sequence_a, sequence_b, chose_left, chose_right, model.
    Archive structure: study_result_1/comp_result_1/data.txt, study_result_2/..., etc.
    data.txt is the component result payload (e.g. jsPsych JSON array of trials).
    """
    rows: List[Dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        # List study_result_* dirs (one per participant run), sorted by numeric id
        study_prefixes = sorted(
            set(
                n.split("/")[0] for n in zf.namelist()
                if re.match(r"study_result_\d+/", n)
            ),
            key=lambda s: int(m.group(0)) if (m := re.search(r"\d+", s)) else 0,
        )
        participant_order = study_prefixes
        for participant_id, study_prefix in enumerate(participant_order):
            # Find data.txt under this study (e.g. study_result_1/comp_result_1/data.txt)
            prefix_with_slash = study_prefix + "/"
            for name in zf.namelist():
                if not name.startswith(prefix_with_slash) or name == prefix_with_slash:
                    continue
                if not name.endswith("data.txt"):
                    continue
                try:
                    data_bytes = zf.read(name)
                    data_str = data_bytes.decode("utf-8", errors="replace").strip()
                    if not data_str:
                        continue
                    data = json.loads(data_str)
                except (json.JSONDecodeError, KeyError):
                    continue
                # jsPsych sends array of trial objects; or single object with trials
                trials = _extract_trials(data)
                for i, trial in enumerate(trials):
                    if "sequence_a" not in trial or "sequence_b" not in trial:
                        continue
                    chose_left = trial.get("chose_left")
                    if chose_left is None:
                        continue
                    rows.append({
                        "participant_id": participant_id,
                        "trial_index": i,
                        "sequence_a": str(trial["sequence_a"]),
                        "sequence_b": str(trial["sequence_b"]),
                        "chose_left": int(bool(chose_left)),
                        "chose_right": 1 - int(bool(chose_left)),
                        "model": "",
                    })
                break  # one component result per study result for our design
    return rows


def _extract_trials(data: Any) -> List[Dict[str, Any]]:
    """Return a list of trial objects from jsPsych/JATOS result data."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Some exports wrap in a key
        if "trials" in data and isinstance(data["trials"], list):
            return data["trials"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        # Single trial as object
        if "sequence_a" in data and "sequence_b" in data:
            return [data]
    return []
