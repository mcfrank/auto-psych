# name: yes_rate_trial_slope
# description: OLS slope of yes response on trial_index (per 100 trials) across all trials; observed below null means people say yes less as the session goes on (a fatigue/criterion drift the stationary model cannot produce), above null means rising yes rate.
def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy() / 100.0
    y = df["rating"].astype(float).to_numpy()
    xc = x - x.mean()
    den = (xc ** 2).sum()
    if den == 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / den)
