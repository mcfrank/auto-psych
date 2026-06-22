# name: max_run_effect_equal_length
# description: Covariance between the difference in normalized maximum run length and the choice of the left sequence, for sequences of equal length.
def test_statistic(df):
    subset = df[df["n_a"] == df["n_b"]]
    if len(subset) <= 1:
        return 0.0
    diff_max_run = subset["max_run_norm_a"] - subset["max_run_norm_b"]
    cov = subset["chose_left"].cov(diff_max_run)
    return (
        float(cov)
        if not (df["chose_left"].isna().all() or diff_max_run.isna().all())
        else 0.0
    )
