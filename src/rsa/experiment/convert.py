"""The page's jsPsych data -> rows of the canonical reference-game trial table.

Input is what the page hands its submit hook: ``jsPsych.data.get().json()``,
a JSON array with one object per jsPsych trial (welcome, consent,
instructions, the practice choice, the test choices). Only the test choices
(``task == "rsa_choice"``, ``phase == "test"``) become rows; the practice
trial is not data.

Rows follow the column contract of the external sources
(`src.rsa.ingest.common.COLUMNS`, documented in
`src/pipelines/outer_loop/projects/rsa_reference/data/README.md`), so
`src.rsa.dataset.context_from_row` loads each one and `load_forced_choice`
reads a CSV of them:

* ``participant_id`` is ``<source>:<id>`` (the id is given by the caller, never
  read from the data, which may carry a recruitment-platform id);
* ``objects`` in the design's canonical order, ``display_order`` the object at
  each screen position, ``choice`` the chosen object's canonical index
  (``display_order[response]``; the page's own ``choice`` must agree);
* ``condition`` the design trial's label, or ``catch`` for a catch trial;
* ``feature_names`` the item's word for each column; ``messages`` the
  design's message set, or every column when the design gives none (every
  feature a word, as in pragmods);
* ``framing`` ``one_word``; ``query_detail`` ``word`` or ``mumble_one_word``;
* ``covariates``: ``rt`` (ms), ``choice_position``, ``is_catch``,
  ``catch_target`` and ``catch_correct`` (catch trials), ``bases``,
  ``spec_index``, ``list_index``, ``list_assignment``, ``design_sha256``.

Every row is ``included``: exclusions (e.g. on catch accuracy) are an
analysis decision, made from the covariates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union

import pandas as pd
import tyro

from src.rsa.experiment.design import trial_context
from src.rsa.ingest.common import Trial, check_frame, frame_of

SERIES = "rsa_reference_jspsych"
FRAMING = "one_word"

_PER_TRIAL_KEYS = (
    "phase",
    "trial_number",
    "n_test_trials",
    "condition",
    "is_catch",
    "catch_target",
    "spec_index",
    "item",
    "feature_names",
    "objects",
    "roles",
    "utterance",
    "word",
    "query",
    "messages",
    "display_order",
    "bases",
    "response",
    "rt",
    "list_index",
    "list_seed",
    "list_assignment",
    "design_name",
    "design_sha256",
)


def _records(data: Union[str, Sequence[dict]]) -> List[dict]:
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise ValueError("jsPsych data must be a JSON array of trial objects")
    return list(data)


def choice_records(data: Union[str, Sequence[dict]]) -> List[dict]:
    """The test choice records, in trial-number order (raises on gaps, repeats or mixed lists)."""
    records = [r for r in _records(data) if r.get("task") == "rsa_choice" and r.get("phase") == "test"]
    if not records:
        raise ValueError("no test choices (task 'rsa_choice', phase 'test') in the data")
    for r in records:
        missing = [k for k in _PER_TRIAL_KEYS if k not in r]
        if missing:
            raise ValueError(f"trial {r.get('trial_number')} lacks {missing}")
    for key in ("n_test_trials", "list_index", "design_sha256", "design_name"):
        values = {json.dumps(r[key]) for r in records}
        if len(values) != 1:
            raise ValueError(f"test choices disagree on {key}: {sorted(values)}")
    n = records[0]["n_test_trials"]
    numbers = sorted(r["trial_number"] for r in records)
    if numbers != list(range(n)):
        raise ValueError(f"expected test trials 0..{n - 1} once each, got {numbers}")
    return sorted(records, key=lambda r: r["trial_number"])


def _row(r: dict, *, source: str, participant_id: str, experiment: str, source_file: str) -> Trial:
    ctx = trial_context(r)
    n_obj = len(ctx.objects)
    response = r["response"]
    if not isinstance(response, int) or not 0 <= response < n_obj:
        raise ValueError(f"trial {r['trial_number']}: response {response!r} is not a screen position 0..{n_obj - 1}")
    if sorted(r["display_order"]) != list(range(n_obj)):
        raise ValueError(f"trial {r['trial_number']}: display_order {r['display_order']} is not a permutation")
    choice = r["display_order"][response]
    if "choice" in r and r["choice"] != choice:
        raise ValueError(f"trial {r['trial_number']}: page recorded choice {r['choice']}, display_order gives {choice}")
    covariates = {
        "rt": r["rt"],
        "choice_position": response,
        "is_catch": bool(r["is_catch"]),
        "bases": r["bases"],
        "spec_index": r["spec_index"],
        "list_index": r["list_index"],
        "list_assignment": r["list_assignment"],
        "design_sha256": r["design_sha256"],
    }
    if r["is_catch"]:
        covariates["catch_target"] = r["catch_target"]
        covariates["catch_correct"] = choice == r["catch_target"]
    return Trial(
        source=source,
        participant_id=f"{source}:{participant_id}",
        batch=r["design_name"],
        source_file=source_file,
        series=SERIES,
        experiment=experiment,
        condition=r["condition"],
        trial_index=r["trial_number"],
        item=r["item"],
        feature_names=r["feature_names"],
        objects=r["objects"],
        object_roles=r["roles"],
        display_order=r["display_order"],
        query=r["query"],
        query_detail="word" if r["query"] == "utterance" else "mumble_one_word",
        framing=FRAMING,
        dv="forced_choice",
        messages=list(range(len(ctx.feature_names))) if r["messages"] is None else r["messages"],
        included=True,
        utterance=r["utterance"],
        choice=choice,
        covariates=covariates,
    )


def convert(
    data: Union[str, Sequence[dict]],
    *,
    participant_id: str,
    experiment: str,
    source: str = "auto_psych",
    source_file: str = "",
) -> pd.DataFrame:
    """One participant's jsPsych data as canonical trial rows (one per test choice)."""
    if not participant_id or ":" in participant_id:
        raise ValueError(f"participant_id must be a non-empty id without ':' (got {participant_id!r})")
    if not experiment:
        raise ValueError("experiment label is required")
    rows = [
        _row(r, source=source, participant_id=participant_id, experiment=experiment, source_file=source_file)
        for r in choice_records(data)
    ]
    frame = frame_of(rows)
    check_frame(frame, source)
    return frame


@dataclass
class Args:
    """Convert one participant's jsPsych data (JSON array) into canonical trial rows (CSV)."""

    data: Path
    """JSON file holding jsPsych.data.get().json() (an array of trial objects)."""
    participant_id: str
    """Anonymous participant id (written as <source>:<id>)."""
    experiment: str
    """Experiment label for the `experiment` column."""
    out: Path
    """CSV to write."""
    source: str = "auto_psych"
    """`source` column and participant id prefix."""


def main(args: Args) -> None:
    frame = convert(
        args.data.read_text(encoding="utf-8"),
        participant_id=args.participant_id,
        experiment=args.experiment,
        source=args.source,
        source_file=args.data.name,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {len(frame)} rows to {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
