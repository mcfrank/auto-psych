# name: late_minus_early_yes_rate
# description: Yes rate on the second half of each participant's session (trial_index >= 225) minus the first half; a nonzero observed value against a null near 0 means responding drifts over the session (fatigue/criterion shift), which the stationary model cannot produce.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy().astype(float)
    late = ti >= 225
    return float(r[late].mean() - r[~late].mean())
