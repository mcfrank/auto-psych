# name: late_minus_early_session_yes_rate
# description: Yes-rate on trials in the second half of the session (trial_index >= 225) minus the first half; a nonzero observed vs null (whose mean tracks only stimulus differences) means a within-session drift in response bias (positive = growing liberalness) the stationary model cannot produce.
def test_statistic(df):
    ti = df["trial_index"].astype(int).values
    r = df["rating"].astype(float).values
    a, b = r[ti >= 225], r[ti < 225]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
