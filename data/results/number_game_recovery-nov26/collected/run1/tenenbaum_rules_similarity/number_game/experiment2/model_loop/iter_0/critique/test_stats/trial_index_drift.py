# name: trial_index_drift
# description: Pearson correlation between trial_index and the yes response; observed differing from null (~0) means responding drifts over the session (e.g. growing conservatism), which the stationary model cannot produce.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
