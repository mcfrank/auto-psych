# name: late_minus_early_yes_rate
# description: Yes rate on trials with trial_index >= 225 minus yes rate on trials < 225; observed below null means people say yes less as the session goes on, a drift the stationary model does not produce.
def test_statistic(df):
    r = df["rating"].astype(float).to_numpy()
    t = df["trial_index"].astype(int).to_numpy()
    late = t >= 225
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(r[late].mean() - r[~late].mean())
