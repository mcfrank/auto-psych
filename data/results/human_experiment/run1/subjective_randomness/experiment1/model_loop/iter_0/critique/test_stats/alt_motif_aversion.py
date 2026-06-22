# name: alt_motif_aversion
# description: Rate of choosing the sequence with fewer alternation motifs per item, when alternation rates are similar.
import numpy as np


def test_statistic(df):
    alt_rate_a = df["alt_motifs_a"] / df["n_a"]
    alt_rate_b = df["alt_motifs_b"] / df["n_b"]
    diff_alt = np.abs(alt_rate_a - alt_rate_b) > 0.05
    similar_alts = np.abs(df["p_alts_a"] - df["p_alts_b"]) <= 0.1
    mask = diff_alt & similar_alts
    if not mask.any():
        return 0.5

    fewer_alt_left = alt_rate_a < alt_rate_b
    fewer_alt_right = alt_rate_b < alt_rate_a
    chose_fewer_alt = (
        df.loc[mask & fewer_alt_left, "chose_left"].sum()
        + df.loc[mask & fewer_alt_right, "chose_right"].sum()
    )
    return float(chose_fewer_alt / mask.sum())
