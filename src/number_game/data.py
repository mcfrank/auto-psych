"""The Bigelow & Piantadosi (2016) number-game dataset (Journal of Open
Psychology Data 4(1)): 606 participants x 450 binary ratings of whether a
target belongs with an example set.

`numbergame_data.csv` also carries `hits`, `p` and `H`, summaries over the
whole dataset; only the raw columns are read here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

RAW_COLUMNS = ["set", "id", "rating", "target", "trial"]


def parse_set(text: str) -> Tuple[int, ...]:
    """'3_ 33_ 99' -> (3, 33, 99), in presented order."""
    return tuple(int(x) for x in str(text).split("_"))


def load_responses(path: Path) -> pd.DataFrame:
    responses = pd.read_csv(path, usecols=RAW_COLUMNS)
    responses["set"] = responses["set"].map(parse_set)
    return responses


def trim_least_typical(responses: pd.DataFrame, n_trimmed: int) -> pd.DataFrame:
    """Drop the `n_trimmed` participants with the lowest total typicality:
    log p for a yes and log(1 - p) for a no, p the (set, target) cell's
    fraction of yes over all participants (the paper's trimming)."""
    p = responses.groupby(["set", "target"])["rating"].transform("mean")
    typicality = np.log(np.where(responses["rating"] == 1, p, 1 - p))
    totals = pd.Series(typicality).groupby(responses["id"].to_numpy()).sum()
    dropped = totals.nsmallest(n_trimmed).index
    return responses[~responses["id"].isin(dropped)].reset_index(drop=True)


def cell_counts(responses: pd.DataFrame) -> pd.DataFrame:
    """One row per (set, target): `yes` count and `n` responses."""
    return (
        responses.groupby(["set", "target"], sort=True)["rating"]
        .agg(yes="sum", n="count")
        .reset_index()
    )
