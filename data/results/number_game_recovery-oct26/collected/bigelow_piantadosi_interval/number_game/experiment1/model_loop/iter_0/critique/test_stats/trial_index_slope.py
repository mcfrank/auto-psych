# name: trial_index_slope
# description: Correlation between trial_index and yes response; nonzero observed vs a ~0 null means the yes-rate drifts over the session (fatigue/learning), which the stationary model cannot produce (sign gives drift direction).
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
