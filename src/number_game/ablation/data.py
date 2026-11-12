"""All three dataset runs' responses in one file: the data the ablation is scored on.

    python -m src.number_game.ablation.data --out <responses.csv>

Each run's cumulative experiment-3 ``model_loop/responses.csv`` of the dataset
run (``data/results/number_game_dataset-oct26/``), in run order: 22,680
responses, the same file as the paper's pooled fits (Figures 2 and 3;
``scripts.paper.number_game_figure_data.pool_responses``). Not the full
Bigelow & Piantadosi bank. The runs hide participant ids, so nothing is
renumbered. An existing file is kept only if it is that file.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

COLLECTED = here() / "data/results/number_game_dataset-oct26/collected"
RUNS = ("run1", "run2", "run3")
COLUMNS = ("number_set", "target", "trial_index", "rating")


def run_responses(run: str) -> Path:
    return COLLECTED / run / "number_game/experiment3/model_loop/responses.csv"


def pooled_text() -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for run in RUNS:
        with run_responses(run).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != COLUMNS:
                raise ValueError(f"{run}: {run_responses(run)} has columns {reader.fieldnames}, not {list(COLUMNS)}")
            writer.writerows(reader)
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
        print(f"[data] {args.out} is up to date")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(args.out, text)
    print(f"[data] {args.out}: {text.count(chr(10)) - 1} responses")


if __name__ == "__main__":
    main(tyro.cli(Args))
