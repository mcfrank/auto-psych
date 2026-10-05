"""Bigelow & Piantadosi (2016)'s fit statistic."""

from __future__ import annotations

import numpy as np
import pandas as pd


def binned_mean_r2(p_model: np.ndarray, cells: pd.DataFrame, width: float = 0.05) -> float:
    """The paper's R^2_mean (Figure 2): bin (set, target) cells by model
    probability in bins of `width`, and square the correlation of each bin's
    position with its mean human p(yes) over cells (`yes / n`)."""
    frame = pd.DataFrame({
        "bin": np.minimum(np.floor(p_model / width), 1 / width - 1),
        "human": cells["yes"].to_numpy() / cells["n"].to_numpy(),
    })
    means = frame.groupby("bin")["human"].mean()
    return float(np.corrcoef(means.index.to_numpy(), means.to_numpy())[0, 1] ** 2)
