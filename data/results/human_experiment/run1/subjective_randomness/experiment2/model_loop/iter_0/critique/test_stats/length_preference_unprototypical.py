# name: length_preference_unprototypical
# description: Proportion of trials where the longer sequence is chosen, conditioned on both sequences being highly unprototypical (p_alts < 0.2 or p_alts > 0.8) and having different lengths.
def test_statistic(df):
    subset = df[
        ((df["p_alts_a"] < 0.2) | (df["p_alts_a"] > 0.8))
        & ((df["p_alts_b"] < 0.2) | (df["p_alts_b"] > 0.8))
        & (df["n_a"] != df["n_b"])
    ]

    if len(subset) == 0:
        return 0.0

    chose_longer = ((subset["n_a"] > subset["n_b"]) & (subset["chose_left"] == 1)) | (
        (subset["n_b"] > subset["n_a"]) & (subset["chose_left"] == 0)
    )

    return float(chose_longer.mean())
