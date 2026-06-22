# name: beta_rep_motifs
# description: Linear regression slope of chose_left on the difference in repeating motifs (rep_motifs_b - rep_motifs_a).

import numpy as np


def test_statistic(df):
    x = df["rep_motifs_b"] - df["rep_motifs_a"]
    y = df["chose_left"]
    if np.var(x) < 1e-8:
        return 0.0
    return float(np.cov(x, y)[0, 1] / np.var(x))
