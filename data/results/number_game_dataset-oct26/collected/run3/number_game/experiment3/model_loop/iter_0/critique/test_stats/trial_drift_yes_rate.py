# name: trial_drift_yes_rate
# description: Yes-rate on trials in the second half of a session (trial_index >= median) minus the first half; observed > null means people become more liberal over the session, a drift the stationary model does not produce.
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    late = ti >= np.median(ti)
    if late.all() or (~late).all():
        return 0.0
    return float(y[late].mean() - y[~late].mean())
