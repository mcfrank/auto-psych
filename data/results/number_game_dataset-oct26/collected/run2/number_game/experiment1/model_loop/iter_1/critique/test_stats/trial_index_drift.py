# name: trial_index_drift
# description: Pearson correlation between trial_index and the yes response across all trials; a nonzero observed value vs a near-zero model value means the yes tendency drifts over the session (fatigue/learning), which the stationary model cannot produce.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    if x.std() == 0 or r.std() == 0:
        return 0.0
    return float(np.corrcoef(x, r)[0, 1])
