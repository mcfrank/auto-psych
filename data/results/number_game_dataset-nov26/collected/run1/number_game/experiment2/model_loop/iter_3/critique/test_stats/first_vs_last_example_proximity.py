# name: first_vs_last_example_proximity
# description: In sets with >=2 examples, among targets within 6 of exactly one example (and >6 from all others), not equal to it, yes-rate when that nearby example is the first shown minus when it is the last shown; the model weights all examples equally, so observed > null means a primacy effect (first example anchors the concept) and observed < null a recency effect the model misses.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    d = np.abs(t - e).min(1)
    nodig = ~((np.mod(t, 10) == np.mod(e, 10)) & pres).any(1)
    mult = (pres & (e1 > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    divi = (pres & (np.mod(e1, t) == 0) & (t != e)).any(1)
    g = np.gcd(t.astype(int), e1.astype(int))
    kin = (pres & (g >= 3)).any(1)

    near = pres & (np.abs(t - e) <= 6) & (t != e)
    one = near.sum(1) == 1
    isex = (pres & (t == e)).any(1)
    idx = np.argmax(near, axis=1)
    base = (n >= 2) & one & ~isex
    a = base & (idx == 0)
    b = base & (idx == n - 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
