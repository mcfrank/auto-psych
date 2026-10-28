# name: session_drift_late_minus_early
# description: Yes-rate on trials in the last third of the session (trial_index >= 300) minus the first third (trial_index < 150); observed above the null means people drift toward "yes" over the session, a nonstationarity the stationary model cannot produce (below means drift toward "no").
def test_statistic(df):
    ti = df["trial_index"].astype(float).values
    y = df["rating"].astype(float).values
    late = ti >= 300
    early = ti < 150
    if late.sum() == 0 or early.sum() == 0:
        return 0.0
    return float(y[late].mean() - y[early].mean())
