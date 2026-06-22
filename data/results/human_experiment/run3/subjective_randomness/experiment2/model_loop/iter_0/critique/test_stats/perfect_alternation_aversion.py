import numpy as np
import pandas as pd


# name: perfect_alternation_aversion
# description: The average choice rate for sequence A when sequence A has perfect alternation (p_alts=1) and sequence B does not, testing if deterministic alternation is penalized more heavily than linear distance from an ideal.
def test_statistic(df):
    mask = (df["p_alts_a"] == 1.0) & (df["p_alts_b"] < 1.0)
    if mask.sum() == 0:
        return np.nan
    return df.loc[mask, "chose_left"].mean()
