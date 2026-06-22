# name: rep_motifs_aversion
# description: Proportion of trials where the sequence with a lower density of repeated motifs (rep_motifs / n) is chosen, conditioned on similar alternation rates and imbalance.
import numpy as np


def test_statistic(df):
    dens_a = df["rep_motifs_a"] / df["n_a"]
    dens_b = df["rep_motifs_b"] / df["n_b"]
    mask = (
        (np.abs(df["p_alts_a"] - df["p_alts_b"]) <= 0.1)
        & (np.abs(df["imbalance_a"] - df["imbalance_b"]) <= 0.1)
        & (dens_a != dens_b)
    )
    sub = df[mask]
    if len(sub) == 0:
        return np.nan

    chose_less_rep = np.where(dens_a < dens_b, sub["chose_left"], sub["chose_right"])
    return float(np.mean(chose_less_rep))
