# name: cross_trial_residual_carryover
# description: Within-participant lag-1 correlation of response residuals (rating minus the leave-one-out cell mean of other participants on the same number_set and target) between consecutive trials, regardless of whether the set changed; observed above null means a yes on one trial makes the next trial's answer more yes than its stimulus warrants (sequential assimilation the stateless model lacks), below means contrast/alternation.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[0, np.diff(ti) < 0])
    y = df["rating"].astype(float).to_numpy()
    cell = (df["number_set"].astype(str) + "|" + df["target"].astype(str)).to_numpy()
    g = pd.Series(y).groupby(cell)
    s = g.transform("sum").to_numpy(); c = g.transform("count").to_numpy()
    res = np.where(c > 1, y - (s - y) / np.maximum(c - 1, 1), np.nan)
    a, b = res[:-1], res[1:]
    ok = (np.diff(pid) == 0) & np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])
