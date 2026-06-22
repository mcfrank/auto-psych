# name: max_run_length_aversion
# description: Mean choice rate of sequence A when it has a longer maximum run length than B by >1, controlling for similar proportion and alternation rates.
import numpy as np


def test_statistic(df):
    mask = (
        (np.abs(df["p_a"] - df["p_b"]) < 0.2)
        & (np.abs(df["p_alts_a"] - df["p_alts_b"]) < 0.2)
        & (df["max_run_a"] > df["max_run_b"] + 1)
    )
    return float(df.loc[mask, "chose_left"].mean())
