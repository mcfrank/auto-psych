import numpy as np
import pandas as pd


# name: max_run_preference
# description: The response rate for sequence A when sequence A has a smaller normalized maximum run length than sequence B, restricting to trials where imbalance is similar, testing if people penalize long runs beyond basic feature proportions.
def test_statistic(df):
    # Condition on similar imbalance to isolate max_run effect
    mask = (np.abs(df["imbalance_a"] - df["imbalance_b"]) < 0.1) & (
        df["max_run_norm_a"] < df["max_run_norm_b"]
    )
    if mask.sum() == 0:
        return np.nan
    return df.loc[mask, "chose_left"].mean()
