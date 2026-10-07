"""Hold out conditions within each source: the loop's train/test split.

    uv run python -m src.rsa.split --trials data/rsa/combined_trials.csv --out-dir data/rsa/split

PI decision (2026-10-07): hold out conditions *within* papers, not whole
papers, so generalisation is not hostage to one paper's idiosyncrasies.

The unit held out is the unit a model must generalise to:

* in an experiment where each included participant gives one trial (pragmods
  E1-E10, Sikos), an experimental condition: (source, experiment, condition);
* in a multi-trial experiment (the Mayn & Demberg games, pragmods sequences),
  a display within its condition: (source, experiment, condition, display),
  i.e. an item. The same people then appear in train and test, on different
  items.

Within each source, units are shuffled with a seeded generator and held out
until at least ``fraction`` of the source's included forced-choice trials are
in the test set. Every row of a unit goes to the same side (production rows
and excluded rows included), so nothing about a held-out unit is in train.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tyro

DISPLAY_COLUMNS = ("objects", "query", "utterance", "messages")
DEFAULT_FRACTION = 0.2


def _counted(frame: pd.DataFrame) -> pd.Series:
    """Rows the loop fits: included forced-choice trials with a display."""
    return (frame["dv"] == "forced_choice") & frame["included"].astype(bool) & (frame["objects"] != "[]")


def multi_trial_experiments(frame: pd.DataFrame) -> set:
    counted = frame[_counted(frame)]
    per_person = counted.groupby(["experiment", "participant_id"]).size()
    medians = per_person.groupby(level=0).median()
    return set(medians[medians > 1].index)


def unit_keys(frame: pd.DataFrame) -> pd.Series:
    """The held-out unit of every row, as a string key."""
    source = frame["source"] if "source" in frame.columns else pd.Series("pragmods", index=frame.index)
    multi = multi_trial_experiments(frame)
    display_cols = [c for c in DISPLAY_COLUMNS if c in frame.columns]
    display = frame[display_cols].astype(str).agg("|".join, axis=1)
    base = source.astype(str) + "|" + frame["experiment"].astype(str) + "|" + frame["condition"].astype(str)
    return base.where(~frame["experiment"].isin(multi), base + "|" + display)


def split(frame: pd.DataFrame, fraction: float = DEFAULT_FRACTION, seed: int = 0
          ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1): {fraction}")
    units = unit_keys(frame)
    counted = _counted(frame)
    source = frame["source"] if "source" in frame.columns else pd.Series("pragmods", index=frame.index)
    held = set()
    report = {}
    rng = np.random.default_rng(seed)
    for src in sorted(source.unique()):
        in_src = source == src
        sizes = counted[in_src].groupby(units[in_src]).sum()
        sizes = sizes[sizes > 0]  # units with no fitted trials cannot make a test set
        if len(sizes) < 2:
            raise ValueError(f"source {src} has {len(sizes)} unit(s) with trials; cannot hold one out")
        order = list(sizes.index[rng.permutation(len(sizes))])
        total, taken, chosen = int(sizes.sum()), 0, []
        for unit in order:
            if taken >= fraction * total:
                break
            chosen.append(unit)
            taken += int(sizes[unit])
        if len(chosen) == len(sizes):
            raise ValueError(f"source {src}: the split would hold out every unit")
        held.update(chosen)
        report[src] = dict(units=int(len(sizes)), held_out_units=len(chosen), trials=total,
                           held_out_trials=taken, held_out_fraction=round(taken / total, 3))
    test_mask = units.isin(held)
    info = dict(fraction=fraction, seed=seed, sources=report,
                multi_trial_experiments=sorted(multi_trial_experiments(frame)),
                held_out_units=sorted(held))
    return frame[~test_mask].copy(), frame[test_mask].copy(), info


@dataclass
class Args:
    trials: Path
    out_dir: Path
    fraction: float = DEFAULT_FRACTION
    seed: int = 0


def main(args: Args) -> None:
    frame = pd.read_csv(args.trials)
    train, test, info = split(frame, args.fraction, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    info["trials_csv"] = str(args.trials)
    (out / "split.json").write_text(json.dumps(info, indent=1))
    for src, r in info["sources"].items():
        print(f"{src}: held out {r['held_out_units']}/{r['units']} units, "
              f"{r['held_out_trials']}/{r['trials']} trials ({r['held_out_fraction']:.0%})")


if __name__ == "__main__":
    main(tyro.cli(Args))
