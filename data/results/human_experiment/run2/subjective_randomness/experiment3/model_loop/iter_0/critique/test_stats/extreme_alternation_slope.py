# name: extreme_alternation_slope
# description: Correlation between preference for A and difference in alternation rate (B - A) when both sequences are highly alternating (p_alts > 0.6).
def test_statistic(df):
    import numpy as np

    mask = (
        (df["p_alts_a"] > 0.6)
        & (df["p_alts_b"] > 0.6)
        & (df["p_alts_a"] != df["p_alts_b"])
    )
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "p_alts_b"] - df.loc[mask, "p_alts_a"],
            df.loc[mask, "chose_left"],
        )[0, 1]
    )
