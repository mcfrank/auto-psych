import numpy as np
import pandas as pd


# name: periodicity_aversion
# description: The response rate for sequence A when it has lower periodicity than sequence B, restricting to trials with similar imbalance.
def test_statistic(df):
    mask = (np.abs(df["imbalance_a"] - df["imbalance_b"]) < 0.1) & (
        df["periodicity_a"] < df["periodicity_b"]
    )
    if mask.sum() == 0:
        return np.nan
    return df.loc[mask, "chose_left"].mean()
