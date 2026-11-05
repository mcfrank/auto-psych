# name: rule_consistent_distance_corr
# description: In sets of >=3 examples with a shared rule (common final digit; else set gcd>=3; else common parity), correlation between distance to the nearest example and rating among non-example targets that obey the rule; observed > null (flatter/less negative) means people apply the rule all-or-none regardless of magnitude while the model keeps a similarity gradient, observed < null means the gradient is even steeper than the model's.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    dist = np.abs(t - e).min(1)
    dig = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    isex = (t == e).any(1)
    mult = (pres & (e1 > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    divi = (pres & (np.mod(e1, t) == 0) & (t != e)).any(1)
    kin = (pres & (np.gcd(t.astype(int), e1.astype(int)) >= 3)).any(1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    ei = e1.astype(int)
    u = np.where(pres, np.mod(ei, 10), -1)
    umax = np.where(pres, u, -1).max(1)
    umin = np.where(pres, u, 99).min(1)
    same_digit = (umax == umin)
    g = np.gcd.reduce(np.where(pres, ei, 0), axis=1)
    par = np.where(pres, np.mod(ei, 2), -1)
    pmax = par.max(1)
    pmin = np.where(pres, par, 9).min(1)
    same_par = pmax == pmin
    ti = t1.astype(int)
    big = n >= 3
    obey = np.where(same_digit, np.mod(ti, 10) == umax,
           np.where(g >= 3, np.mod(ti, np.maximum(g, 1)) == 0,
           np.where(same_par, np.mod(ti, 2) == pmax, False)))
    m = big & obey & ~isex
    if m.sum() < 5 or np.std(r[m]) == 0 or np.std(dist[m]) == 0:
        return 0.0
    return float(np.corrcoef(dist[m], r[m])[0, 1])
