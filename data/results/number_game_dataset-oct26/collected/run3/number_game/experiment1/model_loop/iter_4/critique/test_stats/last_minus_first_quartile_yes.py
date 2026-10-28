# name: last_minus_first_quartile_yes
# description: Yes-rate on trials in the top quartile of trial_index minus yes-rate in the bottom quartile; the model has no session dynamics (null ~0), so observed above null means people grow more liberal over the session, below means more conservative.
def test_statistic(df):
    t = df["trial_index"].astype(float).values
    r = df["rating"].astype(float).values
    q1, q3 = np.quantile(t, [0.25, 0.75])
    lo, hi = r[t <= q1], r[t > q3]
    if len(lo) == 0 or len(hi) == 0:
        return 0.0
    return float(hi.mean() - lo.mean())
