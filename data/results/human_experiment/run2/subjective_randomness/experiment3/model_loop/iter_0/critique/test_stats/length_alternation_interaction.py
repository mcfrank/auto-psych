# name: length_alternation_interaction
# description: Correlation between preference for A and length difference (A - B) when both sequences are highly alternating (p_alts > 0.6).
def test_statistic(df):
    import numpy as np

    mask = (df["n_a"] != df["n_b"]) & (df["p_alts_a"] > 0.6) & (df["p_alts_b"] > 0.6)
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "n_a"] - df.loc[mask, "n_b"], df.loc[mask, "chose_left"]
        )[0, 1]
    )
