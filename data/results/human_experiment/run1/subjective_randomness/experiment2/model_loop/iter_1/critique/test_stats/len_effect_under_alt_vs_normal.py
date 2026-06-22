# name: len_effect_under_alt_vs_normal
# description: Correlation between n_a and chose_left when Sequence A is highly under-alternating (p_alts < 0.3) but Sequence B is normal (0.4 <= p_alts <= 0.6).
import numpy as np


def test_statistic(df):
    sub = df[(df["p_alts_a"] < 0.3) & df["p_alts_b"].between(0.4, 0.6)]
    if len(sub) < 2 or np.var(sub["n_a"]) < 1e-9 or np.var(sub["chose_left"]) < 1e-9:
        return np.nan
    return float(np.corrcoef(sub["n_a"], sub["chose_left"])[0, 1])
