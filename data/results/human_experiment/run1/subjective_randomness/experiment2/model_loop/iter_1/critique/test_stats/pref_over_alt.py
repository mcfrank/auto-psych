# name: pref_over_alt
# description: Mean rate of choosing sequence A when A is highly over-alternating (p_alts > 0.7) and B is normally alternating (0.4 <= p_alts <= 0.6).
import numpy as np


def test_statistic(df):
    sub = df[(df["p_alts_a"] > 0.7) & df["p_alts_b"].between(0.4, 0.6)]
    if len(sub) == 0:
        return np.nan
    return float(np.mean(sub["chose_left"]))
