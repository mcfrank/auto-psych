# name: trial_index_drift
# description: Pearson correlation between trial_index and the yes response; the model has no time-on-task term so predicts ~0, and an observed value away from null_mean means people drift toward more (positive) or fewer (negative) yeses over the session.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
