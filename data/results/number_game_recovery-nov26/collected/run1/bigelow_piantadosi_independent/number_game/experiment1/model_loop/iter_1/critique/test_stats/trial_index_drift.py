# name: trial_index_drift
# description: Pearson correlation between trial_index and the yes response; the model has no session dynamics so predicts ~0 — observed above null means people say yes more as the session goes on, below null means they grow more conservative.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    if x.std() == 0 or r.std() == 0:
        return 0.0
    return float(np.corrcoef(x, r)[0, 1])
