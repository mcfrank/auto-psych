"""The number-game run's own responses: the data the ablation is scored on.

    python -m src.number_game.ablation.data --out <responses.csv>

Experiment 3's cumulative ``model_loop/responses.csv`` of the dataset run
(``data/results/number_game_dataset-oct26/``): 7,560 responses on 150
example sets, the data the final was selected on. Not the full Bigelow &
Piantadosi bank (user decision, 2026-10-27). An existing file is kept only if
it is that file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

RESPONSES = (here() / "data/results/number_game_dataset-oct26/collected/run1/number_game/experiment3"
             / "model_loop/responses.csv")


@dataclass
class Args:
    out: Path


def main(args: Args) -> None:
    from src.runtime.atomic_files import write_text_atomically

    text = RESPONSES.read_text(encoding="utf-8")
    if args.out.exists():
        if args.out.read_text(encoding="utf-8") != text:
            raise ValueError(f"{args.out} exists and is not {RESPONSES}; use a fresh path")
        print(f"[data] {args.out} is up to date")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(args.out, text)
    print(f"[data] {args.out}: {text.count(chr(10)) - 1} responses")


if __name__ == "__main__":
    main(tyro.cli(Args))
