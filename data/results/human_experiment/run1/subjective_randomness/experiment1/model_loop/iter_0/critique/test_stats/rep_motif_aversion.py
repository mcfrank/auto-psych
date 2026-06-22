# name: rep_motif_aversion
# description: Rate of choosing the sequence with fewer repetition motifs per item, when alternation rates are similar.
import numpy as np


def test_statistic(df):
    rep_rate_a = df["rep_motifs_a"] / df["n_a"]
    rep_rate_b = df["rep_motifs_b"] / df["n_b"]
    diff_rep = np.abs(rep_rate_a - rep_rate_b) > 0.05
    similar_alts = np.abs(df["p_alts_a"] - df["p_alts_b"]) <= 0.1
    mask = diff_rep & similar_alts
    if not mask.any():
        return 0.5

    fewer_rep_left = rep_rate_a < rep_rate_b
    fewer_rep_right = rep_rate_b < rep_rate_a
    chose_fewer_rep = (
        df.loc[mask & fewer_rep_left, "chose_left"].sum()
        + df.loc[mask & fewer_rep_right, "chose_right"].sum()
    )
    return float(chose_fewer_rep / mask.sum())
