# name: trial_index_drift
# description: Pearson correlation between trial_index and the yes response across all trials; observed above (below) null means people drift toward saying yes (no) later in the session, a drift the stationary model cannot produce.
def test_statistic(df):
    x = df["trial_index"].astype(float).values
    y = df["rating"].astype(float).values
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
