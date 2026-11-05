# name: five_ending_after_zero_examples
# description: In sets with at least one example ending in 0 and none ending in 5, yes-rate for targets ending in 5 minus yes-rate for targets ending in 1-4 or 6-9 that share no example's final digit; observed > null means people extend round examples to the half-round class (multiples of 5), which the model's 0-ending-only round route and digit dims miss.
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
    digmatch = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(r[a].mean() - r[b].mean())
    has0 = ((np.mod(e1, 10) == 0) & pres).any(1)
    has5 = ((np.mod(e1, 10) == 5) & pres).any(1)
    s = has0 & ~has5
    u = np.mod(t1, 10)
    return diff(s & (u == 5), s & (u != 5) & (u != 0) & ~digmatch)
