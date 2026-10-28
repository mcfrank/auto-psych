# name: within_session_yes_drift
# description: Yes rate on trials in the second half of each session (trial_index >= 225) minus the first half; nonzero observed vs null ~0 means participants become more (positive) or less (negative) liberal over the session, which the stationary model cannot produce.
def test_statistic(df):
    t = df["trial_index"].values
    r = df["rating"].values.astype(float)
    late = t >= 225
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(r[late].mean() - r[~late].mean())
