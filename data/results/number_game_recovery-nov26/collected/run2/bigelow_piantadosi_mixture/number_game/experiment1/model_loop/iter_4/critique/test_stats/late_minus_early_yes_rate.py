# name: late_minus_early_yes_rate
# description: Yes rate in the last third of trials (by trial_index) minus the first third; the model has no time dependence, so observed above null means people become more liberal over the session, below null more conservative.
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    lo, hi = np.quantile(ti, [1/3, 2/3])
    a = y[ti <= lo]; b = y[ti >= hi]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(b.mean() - a.mean())
