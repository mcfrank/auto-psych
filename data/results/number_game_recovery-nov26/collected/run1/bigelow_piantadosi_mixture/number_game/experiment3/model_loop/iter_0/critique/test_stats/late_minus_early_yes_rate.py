# name: late_minus_early_yes_rate
# description: Yes rate on trials with trial_index >= 225 minus yes rate on earlier trials; a nonzero observed value beyond the model's means the model (no time dependence) misses a drift in willingness to say yes over the session.
def test_statistic(df):
    t = df["trial_index"].to_numpy()
    r = df["rating"].astype(float).to_numpy()
    late = t >= 225
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(r[late].mean() - r[~late].mean())
