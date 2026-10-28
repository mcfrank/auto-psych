# name: early_minus_late_yes
# description: Yes-rate on trials with trial_index < 90 minus yes-rate on trials with trial_index >= 90; observed above null means people grow more conservative over the session, a drift the stationary model cannot produce.
def test_statistic(df):
    ti = df["trial_index"].values
    r = df["rating"].values.astype(float)
    e = ti < 90
    if e.sum() == 0 or (~e).sum() == 0:
        return 0.0
    return float(r[e].mean() - r[~e].mean())
