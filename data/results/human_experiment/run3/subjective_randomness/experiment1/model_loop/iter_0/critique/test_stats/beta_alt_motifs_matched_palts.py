# name: beta_alt_motifs_matched_palts
# description: Slope of chose_left on (alt_motifs_b - alt_motifs_a) for trials where the absolute difference in p_alts is less than 0.1.

import numpy as np


def test_statistic(df):
    mask = abs(df["p_alts_a"] - df["p_alts_b"]) < 0.1
    if mask.sum() < 2:
        return 0.0
    x = (df["alt_motifs_b"] - df["alt_motifs_a"])[mask]
    y = df["chose_left"][mask]
    if np.var(x) < 1e-8:
        return 0.0
    return float(np.cov(x, y)[0, 1] / np.var(x))
