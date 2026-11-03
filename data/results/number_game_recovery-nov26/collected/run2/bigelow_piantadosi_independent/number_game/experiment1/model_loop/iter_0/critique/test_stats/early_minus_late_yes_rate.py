# name: early_minus_late_yes_rate
# description: Yes rate in the first third of trial_index minus yes rate in the last third; a nonzero t_observed vs null_mean (~0) means response bias drifts over the session, which the model cannot produce.
def test_statistic(df):
    ti = df["trial_index"].astype(float)
    lo, hi = ti.quantile(1/3), ti.quantile(2/3)
    r = df["rating"].astype(float)
    a, b = r[ti <= lo], r[ti >= hi]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
