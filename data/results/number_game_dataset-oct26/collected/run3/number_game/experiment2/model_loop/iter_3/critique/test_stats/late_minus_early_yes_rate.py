# name: late_minus_early_yes_rate
# description: Yes-rate on trials with trial_index >= 300 minus yes-rate on trials with trial_index < 150; observed > model means people drift toward "yes" over the session more than the trial-invariant model predicts (observed < model means drift toward "no").
def test_statistic(df):
    t = df["trial_index"].astype(float).values
    r = df["rating"].astype(float).values
    late = t >= 300
    early = t < 150
    if late.sum() == 0 or early.sum() == 0:
        return 0.0
    return float(r[late].mean() - r[early].mean())
