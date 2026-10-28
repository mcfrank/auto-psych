# name: trial_index_drift
# description: Pearson correlation between trial_index and yes (0/1) across all trials; the model has no session dynamics, so observed away from null means people drift toward more yes (positive) or more no (negative) as the session proceeds.
def test_statistic(df):
    x = df["trial_index"].to_numpy(float)
    y = df["rating"].to_numpy(float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
