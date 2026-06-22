# name: periodicity_aversion
# description: Mean choice rate of sequence A when it is more periodic than B, controlling for similar proportion and alternation rates.
import numpy as np


def test_statistic(df):
    mask = (
        (np.abs(df["p_a"] - df["p_b"]) < 0.2)
        & (np.abs(df["p_alts_a"] - df["p_alts_b"]) < 0.2)
        & (df["periodicity_a"] > df["periodicity_b"] + 0.2)
    )
    return float(df.loc[mask, "chose_left"].mean())
