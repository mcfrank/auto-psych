# name: length_difference_saturation
# description: Difference in preference for the longer sequence between trials with a large length difference (>=4) and trials with a small length difference (<=2).
def test_statistic(df):
    subset = df[
        (abs(df["imbalance_a"] - df["imbalance_b"]) < 0.2)
        & (abs(df["p_alts_a"] - df["p_alts_b"]) < 0.2)
    ]
    diff_n = abs(subset["n_a"] - subset["n_b"])

    small_diff = subset[diff_n <= 2]
    large_diff = subset[diff_n >= 4]

    if len(small_diff) == 0 or len(large_diff) == 0:
        return 0.0

    def prob_longer(d):
        chose_longer = ((d["n_a"] > d["n_b"]) & (d["chose_left"] == 1)) | (
            (d["n_b"] > d["n_a"]) & (d["chose_left"] == 0)
        )
        return chose_longer.mean()

    return float(prob_longer(large_diff) - prob_longer(small_diff))
