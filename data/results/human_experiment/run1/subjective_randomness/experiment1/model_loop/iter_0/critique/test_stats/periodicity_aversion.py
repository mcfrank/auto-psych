# name: periodicity_aversion
# description: Rate of choosing the sequence with lower periodicity when periodicities differ.
import numpy as np


def test_statistic(df):
    diff_per = df["periodicity_a"] != df["periodicity_b"]
    mask = diff_per
    if not mask.any():
        return 0.5

    lower_per_left = df["periodicity_a"] < df["periodicity_b"]
    lower_per_right = df["periodicity_b"] < df["periodicity_a"]
    chose_lower_per = (
        df.loc[mask & lower_per_left, "chose_left"].sum()
        + df.loc[mask & lower_per_right, "chose_right"].sum()
    )
    return float(chose_lower_per / mask.sum())
