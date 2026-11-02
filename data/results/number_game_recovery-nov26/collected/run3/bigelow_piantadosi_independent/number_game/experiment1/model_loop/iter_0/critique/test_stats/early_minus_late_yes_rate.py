# name: early_minus_late_yes_rate
# description: Yes-rate on trials with trial_index < 90 minus yes-rate on later trials; observed > null means people say yes more early in the session than the static model predicts (a time-varying response bias the model lacks).
def test_statistic(df):
    r = df["rating"].astype(float).to_numpy()
    ti = df["trial_index"].astype(int).to_numpy()
    e = ti < 90
    if e.sum() == 0 or (~e).sum() == 0:
        return 0.0
    return float(r[e].mean() - r[~e].mean())
