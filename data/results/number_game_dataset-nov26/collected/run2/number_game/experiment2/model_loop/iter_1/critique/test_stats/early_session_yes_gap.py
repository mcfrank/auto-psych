# name: early_session_yes_gap
# description: Yes-rate in the first 90 trials of a session (trial_index < 90) minus yes-rate in later trials; the model has no time dependence, so observed below null means people start more conservative and become more liberal over the session (and vice versa).
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    a = ti < 90
    if a.sum() == 0 or (~a).sum() == 0:
        return 0.0
    return float(r[a].mean() - r[~a].mean())
