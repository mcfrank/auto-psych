# name: trial_index_drift_corr
# description: Correlation between trial_index and the yes response across all trials; observed above/below null means people's yes-rate drifts up/down over the session, which the stationary model cannot produce.
def test_statistic(df):
    r = df["rating"].to_numpy().astype(float); ti = df["trial_index"].to_numpy().astype(float)
    if r.std() == 0 or ti.std() == 0:
        return 0.0
    return float(np.corrcoef(r, ti)[0, 1])
