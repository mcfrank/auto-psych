# name: late_minus_early_yes_rate
# description: Yes rate on trials in the last third of the session (trial_index >= 300) minus the first third (trial_index < 150); nonzero observed vs a near-zero null means a drift in willingness to say yes over the session that the stationary model cannot produce (positive = people become more liberal over time).
def test_statistic(df):
    ti = df["trial_index"].values
    r = df["rating"].values.astype(float)
    late, early = ti >= 300, ti < 150
    if late.sum() == 0 or early.sum() == 0:
        return 0.0
    return float(r[late].mean() - r[early].mean())
