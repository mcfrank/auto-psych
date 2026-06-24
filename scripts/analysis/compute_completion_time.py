"""CLI: mean (and median/spread) participant completion time for the human studies.

The collected ``responses.csv`` files do NOT contain timing — the pipeline drops
jsPsych's per-trial ``rt``/``time_elapsed`` when it writes the CSV
(``_rows_from_trial_data`` in ``src/pipelines/outer_loop/collect.py``). The only
authoritative per-participant completion time is Prolific's ``time_taken`` field
(seconds between a participant starting and submitting the study), so this script
pulls submissions straight from the Prolific API.

You must supply the real Prolific study IDs — they are redacted to
``[REDACTED_PROLIFIC_ID]`` everywhere in the repo. The API token is read by the
existing client (``PROLIFIC_API_TOKEN`` env var or ``.secrets``).

Usage:
    # One study.
    uv run python scripts/analysis/compute_completion_time.py --study-ids 64f...abc

    # Several studies (e.g. all 9 run x experiment combos), with labels for the
    # per-study breakdown. ``labels`` must line up 1:1 with ``study-ids``.
    uv run python scripts/analysis/compute_completion_time.py \\
        --study-ids 64f...a 64f...b 64f...c \\
        --labels run1/e1 run1/e2 run1/e3 \\
        --output-csv data/results/human_experiment/completion_times.csv
"""

from __future__ import annotations

import csv
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.runtime.prolific import list_submissions  # noqa: E402

# Statuses Prolific assigns to a submission that actually reached the end of the
# study. We exclude RETURNED / TIMED-OUT / REJECTED — those participants did not
# complete it, so their (often missing) time_taken would distort the mean.
DEFAULT_STATUSES = ("APPROVED", "AWAITING REVIEW", "PARTIALLY APPROVED")


@dataclass(frozen=True)
class CompletionRecord:
    """One participant's completion time within a study."""

    study_id: str
    label: str
    participant_id: str
    seconds: float


@dataclass
class Args:
    """Compute mean participant completion time from Prolific ``time_taken``."""

    study_ids: tuple[str, ...] = ()
    """Real Prolific study IDs (the repo only stores redacted placeholders)."""
    labels: tuple[str, ...] = ()
    """Optional human-readable label per study (e.g. ``run1/e3``); must match
    ``study_ids`` 1:1 when given. Defaults to the study IDs themselves."""
    statuses: tuple[str, ...] = DEFAULT_STATUSES
    """Submission statuses counted as "completed"."""
    output_csv: Optional[Path] = None
    """If set, write one row per participant (study, label, participant, time)."""


def completion_time_statistics(seconds: Sequence[float]) -> "OrderedDict[str, float]":
    """Descriptive statistics for a list of completion times (in seconds)."""
    if len(seconds) == 0:
        raise ValueError("no completion times to summarize")
    arr = np.asarray(seconds, dtype=float)
    return OrderedDict(
        [
            ("n", int(arr.size)),
            ("mean_seconds", float(arr.mean())),
            ("median_seconds", float(np.median(arr))),
            ("std_seconds", float(arr.std(ddof=1)) if arr.size > 1 else 0.0),
            ("min_seconds", float(arr.min())),
            ("max_seconds", float(arr.max())),
            ("mean_minutes", float(arr.mean()) / 60.0),
            ("median_minutes", float(np.median(arr)) / 60.0),
        ]
    )


def time_taken_for_study(
    study_id: str,
    statuses: Sequence[str],
    label: Optional[str] = None,
) -> list[CompletionRecord]:
    """Completion records for the finished submissions of one study.

    Fails loudly if the API call errors, or if a submission in a "completed"
    status has no ``time_taken`` (which would mean we are about to silently
    drop a participant who really finished).
    """
    submissions, err = list_submissions(study_id)
    if err is not None:
        raise RuntimeError(f"Prolific list_submissions({study_id!r}) failed: {err}")
    if submissions is None:
        raise RuntimeError(f"Prolific returned no submissions for study {study_id!r}")

    label = label if label is not None else study_id
    records: list[CompletionRecord] = []
    for sub in submissions:
        if sub.get("status") not in statuses:
            continue
        time_taken = sub.get("time_taken")
        if time_taken is None:
            raise ValueError(
                f"submission {sub.get('id')!r} in study {study_id!r} has status "
                f"{sub.get('status')!r} but no time_taken"
            )
        participant_id = sub.get("participant_id") or sub.get("participant") or ""
        records.append(
            CompletionRecord(study_id, label, str(participant_id), float(time_taken))
        )
    return records


def summarize(
    study_ids: Sequence[str],
    labels: Sequence[str],
    statuses: Sequence[str],
) -> dict:
    """Pool completion times across studies and break them down per study.

    Returns ``{"records", "per_study", "overall"}``. ``per_study`` maps each
    label to its statistics; ``overall`` pools every participant.
    """
    if not study_ids:
        raise ValueError("no study_ids given")

    records: list[CompletionRecord] = []
    per_study: "OrderedDict[str, OrderedDict[str, float]]" = OrderedDict()
    for study_id, label in zip(study_ids, labels):
        study_records = time_taken_for_study(study_id, statuses, label=label)
        records.extend(study_records)
        per_study[label] = completion_time_statistics(
            [r.seconds for r in study_records]
        )

    overall = completion_time_statistics([r.seconds for r in records])
    return {"records": records, "per_study": per_study, "overall": overall}


def _resolve_labels(study_ids: Sequence[str], labels: Sequence[str]) -> tuple[str, ...]:
    if not labels:
        return tuple(study_ids)
    if len(labels) != len(study_ids):
        raise ValueError(
            f"got {len(labels)} labels for {len(study_ids)} study_ids; "
            "they must line up 1:1"
        )
    return tuple(labels)


def _print_statistics(title: str, stats: "OrderedDict[str, float]") -> None:
    print(title)
    print(f"  n               : {stats['n']}")
    print(
        f"  mean            : {stats['mean_minutes']:.2f} min "
        f"({stats['mean_seconds']:.0f} s)"
    )
    print(
        f"  median          : {stats['median_minutes']:.2f} min "
        f"({stats['median_seconds']:.0f} s)"
    )
    print(f"  std             : {stats['std_seconds'] / 60.0:.2f} min")
    print(
        f"  min / max       : {stats['min_seconds'] / 60.0:.2f} / "
        f"{stats['max_seconds'] / 60.0:.2f} min"
    )


def _write_csv(path: Path, records: Sequence[CompletionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["study_id", "label", "participant_id", "time_taken_seconds", "time_taken_minutes"]
        )
        for r in records:
            writer.writerow(
                [r.study_id, r.label, r.participant_id, f"{r.seconds:.0f}", f"{r.seconds / 60.0:.3f}"]
            )


def main(args: Args) -> None:
    if not args.study_ids:
        raise ValueError(
            "no --study-ids given. The repo only stores redacted placeholders; "
            "pass the real Prolific study IDs (one per run x experiment)."
        )
    labels = _resolve_labels(args.study_ids, args.labels)

    result = summarize(args.study_ids, labels, args.statuses)

    if len(result["per_study"]) > 1:
        for label, stats in result["per_study"].items():
            _print_statistics(f"[{label}]", stats)
            print()

    _print_statistics("Overall (all participants pooled)", result["overall"])

    if args.output_csv is not None:
        _write_csv(args.output_csv, result["records"])
        print(f"\nWrote {len(result['records'])} per-participant rows to {args.output_csv}")


if __name__ == "__main__":
    main(tyro.cli(Args))
