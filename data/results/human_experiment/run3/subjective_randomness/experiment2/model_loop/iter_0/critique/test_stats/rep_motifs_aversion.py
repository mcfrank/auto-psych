import numpy as np
import pandas as pd


# name: rep_motifs_aversion
# description: The response rate for sequence A when it has fewer repeating motifs than sequence B, controlling for similar imbalance and alternation rates.
def test_statistic(df):
    mask = (
        (np.abs(df["imbalance_a"] - df["imbalance_b"]) < 0.1)
        & (np.abs(df["p_alts_a"] - df["p_alts_b"]) < 0.1)
        & (df["rep_motifs_a"] < df["rep_motifs_b"])
    )
    if mask.sum() == 0:
        return np.nan
    return df.loc[mask, "chose_left"].mean()
