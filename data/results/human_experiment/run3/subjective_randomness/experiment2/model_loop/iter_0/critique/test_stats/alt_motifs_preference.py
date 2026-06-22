import numpy as np
import pandas as pd


# name: alt_motifs_preference
# description: The response rate for sequence A when it has more alternating motifs than sequence B, conditional on similar overall bigram alternation rates.
def test_statistic(df):
    mask = (np.abs(df["p_alts_a"] - df["p_alts_b"]) < 0.1) & (
        df["alt_motifs_a"] > df["alt_motifs_b"]
    )
    if mask.sum() == 0:
        return np.nan
    return df.loc[mask, "chose_left"].mean()
