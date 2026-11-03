# name: early_session_yes_excess
# description: Yes rate on trials with trial_index < 90 minus yes rate on later trials; observed > null means people say yes more early in the session than the (time-invariant) model reproduces.
def test_statistic(df):
    ti = df["trial_index"].astype(float).values
    r = df["rating"].astype(float).values
    early = ti < 90
    if early.sum() == 0 or (~early).sum() == 0:
        return 0.0
    return float(r[early].mean() - r[~early].mean())
