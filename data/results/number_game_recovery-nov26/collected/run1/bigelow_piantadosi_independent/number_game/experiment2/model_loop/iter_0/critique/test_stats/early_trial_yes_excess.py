# name: early_trial_yes_excess
# description: Yes rate on trials with trial_index < 90 minus yes rate on later trials; observed > null means people say yes more early in the session (a drift the stationary model cannot produce).
def test_statistic(df):
    y = df["rating"].astype(float).to_numpy()
    early = df["trial_index"].to_numpy() < 90
    if early.sum() == 0 or (~early).sum() == 0:
        return 0.0
    return float(y[early].mean() - y[~early].mean())
