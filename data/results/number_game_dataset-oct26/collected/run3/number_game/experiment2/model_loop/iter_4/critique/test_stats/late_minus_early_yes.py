# name: late_minus_early_yes
# description: Yes-rate on trials in the top third of trial_index minus the bottom third; nonzero observed vs a ~0 null means the static model misses a drift in willingness to say yes over the session.
def test_statistic(df):
    ti = df["trial_index"].astype(float).values
    r = df["rating"].astype(float).values
    q1, q2 = np.quantile(ti, [1/3, 2/3])
    a = r[ti >= q2]; b = r[ti <= q1]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
