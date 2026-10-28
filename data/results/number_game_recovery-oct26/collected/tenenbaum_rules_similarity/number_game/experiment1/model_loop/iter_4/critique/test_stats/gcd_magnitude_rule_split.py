# name: gcd_magnitude_rule_split
# description: For sets of 2+ examples with greatest common divisor g in 3-12 (g != 10), yes rate on non-example multiples of g when g >= 7 minus when g is 3-6; the kind-pooled prior treats all multiples-of-n rules alike, so observed below null_mean means people trust large-divisor rules (7, 9, 11, 12) less relative to small ones than the model predicts, above means more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n_ex = valid.sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    e = np.where(valid, ex, 0).astype(np.int64)
    g = e[:, 0]
    for j in range(1, e.shape[1]):
        g = np.gcd(g, e[:, j])
    ok = (n_ex >= 2) & (g >= 3) & (g <= 12) & (g != 10)
    gs = np.maximum(g, 1)
    rule = ok & (~member) & (np.mod(t.astype(np.int64), gs) == 0)
    a = rule & (g >= 7); b = rule & (g <= 6)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
