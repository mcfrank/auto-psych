# name: late_minus_early_yes_rate
# description: Yes-rate on trials with trial_index >= 225 minus yes-rate on trials with trial_index < 225; observed differing from null means a drift in yes-bias across the session that the stationary model does not capture (observed < null: people become more conservative over time).
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].astype(float).to_numpy()
    late = ti >= 225
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(r[late].mean() - r[~late].mean())
