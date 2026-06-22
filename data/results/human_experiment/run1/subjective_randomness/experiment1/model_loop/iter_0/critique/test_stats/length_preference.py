# name: length_preference
# description: The overall rate of choosing the longer sequence when sequence lengths differ.
import numpy as np


def test_statistic(df):
    longer_left = df["n_a"] > df["n_b"]
    longer_right = df["n_b"] > df["n_a"]
    mask = longer_left | longer_right
    if not mask.any():
        return 0.5
    chose_longer = (
        df.loc[longer_left, "chose_left"].sum()
        + df.loc[longer_right, "chose_right"].sum()
    )
    return float(chose_longer / mask.sum())
