# name: trial_index_drift
# description: Pearson correlation between trial_index and rating across all trials; observed differing from null (near 0, the model has no time dependence) means yes-rate drifts over the session (positive = more yes later, negative = fewer).
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
