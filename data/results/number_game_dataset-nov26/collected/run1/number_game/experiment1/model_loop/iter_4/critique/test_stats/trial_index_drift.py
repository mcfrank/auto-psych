# name: trial_index_drift
# description: Pearson correlation between trial_index and the yes response across all trials; observed != null (null ~0, the model has no time dependence) means people drift toward saying yes (positive) or no (negative) over the session.
def test_statistic(df):
    x = df["trial_index"].astype(float).values
    y = df["rating"].astype(float).values
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
