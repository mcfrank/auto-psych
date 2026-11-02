# name: trial_index_yes_drift
# description: Yes rate in the second half of the session (trial_index >= 225) minus the first half; a nonzero observed value relative to null means people's yes bias drifts over the session (positive: more liberal later, negative: more conservative later), which the stationary model cannot produce beyond stimulus-order differences.
def test_statistic(df):
    ti = df["trial_index"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    late = ti >= 225
    if late.all() or (~late).all():
        return 0.0
    return float(y[late].mean() - y[~late].mean())
