# name: trial_index_drift
# description: Pearson correlation between trial_index and yes response across all trials; nonzero observed vs a ~0 null means yes-rate drifts over the session (fatigue/criterion shift) which the stationary model does not produce (sign gives drift direction).
def test_statistic(df):
    t = df["trial_index"].values.astype(float)
    r = df["rating"].values.astype(float)
    if t.std() == 0 or r.std() == 0:
        return 0.0
    return float(np.corrcoef(t, r)[0, 1])
