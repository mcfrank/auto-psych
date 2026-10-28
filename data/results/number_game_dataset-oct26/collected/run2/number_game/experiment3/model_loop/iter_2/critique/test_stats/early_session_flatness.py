# name: early_session_flatness
# description: Yes-rate on trials 50-99 minus trials 0-49; observed < null means the criterion does not relax at all during the first ~100 trials (a delayed, step-like shift) whereas the model's saturating exponential relaxation predicts the steepest rise right at the start.
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    a = (ti >= 50) & (ti < 100)
    b = ti < 50
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
