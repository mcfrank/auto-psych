import numpy as np
import pandas as pd


# name: length_preference_matched_features
# description: The average choice rate for sequence A when it is strictly longer than sequence B, restricting to trials where their imbalance and alternation rates are similar.
def test_statistic(df):
    mask = (
        (df["n_a"] > df["n_b"])
        & (np.abs(df["imbalance_a"] - df["imbalance_b"]) < 0.1)
        & (np.abs(df["p_alts_a"] - df["p_alts_b"]) < 0.1)
    )
    if mask.sum() == 0:
        return np.nan
    return df.loc[mask, "chose_left"].mean()
