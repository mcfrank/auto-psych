# name: length_preference_prototypical
# description: Proportion of trials where the longer sequence is chosen, conditioned on both sequences being highly prototypical (imbalance < 0.2 and 0.4 < p_alts < 0.6) and having different lengths.
def test_statistic(df):
    subset = df[
        (df["imbalance_a"] < 0.2)
        & (df["imbalance_b"] < 0.2)
        & (df["p_alts_a"] > 0.4)
        & (df["p_alts_a"] < 0.6)
        & (df["p_alts_b"] > 0.4)
        & (df["p_alts_b"] < 0.6)
        & (df["n_a"] != df["n_b"])
    ]

    if len(subset) == 0:
        return 0.0

    chose_longer = ((subset["n_a"] > subset["n_b"]) & (subset["chose_left"] == 1)) | (
        (subset["n_b"] > subset["n_a"]) & (subset["chose_left"] == 0)
    )

    return float(chose_longer.mean())
