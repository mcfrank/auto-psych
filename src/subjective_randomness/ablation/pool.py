"""All three runs' final data in one file: the data the ablation is scored on.

    python -m src.subjective_randomness.ablation.pool --out <responses.csv>

Each run's cumulative experiment-3 responses, in run order, each (run,
participant) renumbered in order of first appearance: 23,104 trials of 359
people, the same file as the paper's pooled fits (Table 1;
``scripts.paper.sr_figure_data.pool_responses``). An existing file is kept
only if it is that file.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

COLLECTED = here() / "data/results/human_experiment_sr-oct26/collected"
RUNS = ("run1", "run2", "run3")
COLUMNS = ("sequence_a", "sequence_b", "participant_id", "trial_index", "chose_left")


def run_responses(run: str) -> Path:
    return COLLECTED / run / "subjective_randomness/experiment3/model_loop/responses.csv"


def pooled_text() -> str:
    new_ids: dict = {}
    out = io.StringIO()
    out.write(",".join(COLUMNS) + "\n")
    for run in RUNS:
        with run_responses(run).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != COLUMNS:
                raise ValueError(f"{run}: {run_responses(run)} has columns {reader.fieldnames}, not {list(COLUMNS)}")
            for row in reader:
                row["participant_id"] = str(new_ids.setdefault((run, row["participant_id"]), len(new_ids)))
                out.write(",".join(row[c] for c in COLUMNS) + "\n")
    return out.getvalue()


@dataclass
class Args:
    out: Path


def main(args: Args) -> None:
    from src.runtime.atomic_files import write_text_atomically

    text = pooled_text()
    if args.out.exists():
        if args.out.read_text(encoding="utf-8") != text:
            raise ValueError(f"{args.out} exists and is not the pooled data of {list(RUNS)}; use a fresh path")
        print(f"[pool] {args.out} is up to date")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(args.out, text)
    print(f"[pool] {args.out}: {text.count(chr(10)) - 1} trials")


if __name__ == "__main__":
    main(tyro.cli(Args))
