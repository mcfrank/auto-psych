# name: trial_index_drift
# description: Pearson correlation between trial_index and rating (yes); the model has no time dependence so the null is ~0, observed above null means participants say yes more as the session goes on, below means they become more conservative over trials.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
