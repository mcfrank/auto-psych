# name: late_minus_early_yes_rate
# description: Yes rate in the second half of each session (trial_index >= 225) minus the first half; a nonzero observed value beyond the null means people drift in yes-bias over the session (fatigue/learning), which the stationary model does not produce.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy().astype(float)
    late = ti >= 225
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(r[late].mean() - r[~late].mean())
