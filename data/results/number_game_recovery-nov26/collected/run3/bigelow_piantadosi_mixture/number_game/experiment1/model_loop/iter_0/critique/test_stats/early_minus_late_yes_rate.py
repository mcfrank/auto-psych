# name: early_minus_late_yes_rate
# description: Yes rate on trials with trial_index < 90 minus yes rate on trials with trial_index >= 90; observed > null means people say yes more early in the session than the time-invariant model produces (a drift the model lacks).
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy().astype(float)
    early = ti < 90
    if early.sum() == 0 or (~early).sum() == 0:
        return 0.0
    return float(r[early].mean() - r[~early].mean())
