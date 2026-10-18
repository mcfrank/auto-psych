# name: trial_index_drift
# description: Yes rate in the second half of trial_index minus the first half (median split); observed differing from null (~0) means a session-time drift in yes-bias that the stationary model cannot produce (positive = humans say yes more later on).
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    late = ti > np.median(ti)
    if late.all() or (~late).all():
        return 0.0
    return float(r[late].mean() - r[~late].mean())
