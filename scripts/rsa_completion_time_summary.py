"""Mean and median completion times of the live RSA studies, from the downloaded table.

Reads the per-submission table that ``scripts/rsa_completion_times.py`` writes
(no Prolific IDs) and prints, for completed submissions only, the mean and
median completion time in minutes per run and experiment, per run, and pooled.

    uv run python -m scripts.rsa_completion_time_summary
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro
from pyprojroot import here

from scripts.rsa_completion_times import COMPLETED_STATUSES


@dataclass
class Args:
    table: Path = here("data/rsa/live/completion_times/rsa_completion_times.csv")
    """The per-submission table from scripts/rsa_completion_times.py."""


def summary(minutes: pd.Series) -> pd.DataFrame:
    return minutes.agg(["count", "mean", "median"]).to_frame().T


def main(args: Args):
    df = pd.read_csv(args.table)
    completed = df[df["status"].isin(COMPLETED_STATUSES)].copy()
    if completed["time_taken_seconds"].isna().any():
        raise ValueError(f"{completed['time_taken_seconds'].isna().sum()} completed submissions have no time")
    completed["minutes"] = completed["time_taken_seconds"] / 60

    print(f"{len(completed)} completed submissions of {len(df)} (statuses: {df['status'].value_counts().to_dict()})")
    print("\nCompletion time (minutes), by run and experiment")
    print(completed.groupby(["run", "experiment"])["minutes"].agg(["count", "mean", "median"]).round(2).to_string())
    print("\nBy run")
    print(completed.groupby("run")["minutes"].agg(["count", "mean", "median"]).round(2).to_string())
    print("\nAll runs")
    print(summary(completed["minutes"]).round(2).to_string(index=False))
    chains = completed[completed["run"] != "prereg"]["minutes"]
    print("\nChains c0-c2 pooled")
    print(summary(chains).round(2).to_string(index=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
