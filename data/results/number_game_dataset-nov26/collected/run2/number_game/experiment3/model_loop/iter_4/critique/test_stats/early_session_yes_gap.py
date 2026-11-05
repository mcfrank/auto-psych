# name: early_session_yes_gap
# description: Yes rate on trials with trial_index < 118 (first quarter of the session) minus yes rate on later trials; observed < null means people start out more conservative than later in the session, a warm-up drift the stationary model cannot produce (observed > null: they start more liberal).
def test_statistic(df):
    ti = df["trial_index"].astype(float).values
    r = df["rating"].astype(float).values
    e = ti < 118
    if e.sum() == 0 or (~e).sum() == 0:
        return 0.0
    return float(r[e].mean() - r[~e].mean())
