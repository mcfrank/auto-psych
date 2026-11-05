# name: trial_drift
# description: Correlation between trial_index and response (yes=1) across all trials; observed differing from null means endorsement drifts over the session (e.g. growing liberal/conservative) which the stationary model does not capture.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy(); y = df["rating"].astype(float).to_numpy()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
