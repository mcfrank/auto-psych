# name: trial_index_rating_correlation
# description: Pearson correlation between trial_index and rating across all trials; observed more negative than null means people say yes less often as the session goes on, beyond what the stimulus order implies, and the model has no such drift.
def test_statistic(df):
    x = df["trial_index"].astype(float).values
    y = df["rating"].astype(float).values
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
