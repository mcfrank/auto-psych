# name: late_minus_early_yes
# description: Yes rate on the second half of each session (trial_index >= half the maximum) minus the first half; observed above null means people grow more liberal over the session than the stationary model predicts (below null: more conservative).
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    late = ti >= (ti.max() + 1) / 2.0
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(y[late].mean() - y[~late].mean())
