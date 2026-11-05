# name: first_vs_last_example_proximity
# description: In sets of >=2 examples, for targets 1-9 from exactly one example (all others >=10 away, no shared final digit, not an example), yes-rate when that near example is shown first minus when it is shown last; observed > null means a primacy advantage (the first example anchors generalization) the order-free model lacks, observed < null a recency advantage.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    d_all = np.abs(t - e)
    dist = d_all.min(1)
    dig = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    isex = (t == e).any(1)
    mult = (pres & (e1 > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    divi = (pres & (np.mod(e1, t) == 0) & (t != e)).any(1)
    kin = (pres & (np.gcd(t.astype(int), e1.astype(int)) >= 3)).any(1)
    emin = np.where(pres, ex, 1e9).min(1)
    emax = np.where(pres, ex, -1e9).max(1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    near = pres & (d_all >= 1) & (d_all <= 9)
    one = near.sum(1) == 1
    far_ok = ((d_all >= 10) | near | ~pres).all(1)
    base = (n >= 2) & one & far_ok & ~dig & ~isex
    idx = np.argmax(near, axis=1)
    first = idx == 0
    last = idx == (n - 1)
    return rate(base & first) - rate(base & last)
