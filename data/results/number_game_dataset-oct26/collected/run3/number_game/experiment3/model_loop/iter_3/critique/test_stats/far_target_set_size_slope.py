# name: far_target_set_size_slope
# description: Among non-example targets more than 15 from every example, OLS slope of yes (0/1) on the number of examples (1-4); observed more negative than null means people's far-target acceptance falls with set size faster than the model's evidence-weight and guessing-decline mechanisms predict (observed less negative / positive: falls slower).
def test_statistic(df):
    import math as _m
    sets = df["number_set"].astype(str)
    uniq = pd.Series(sets.unique())
    lists = [[int(x) for x in s.split(",")] for s in uniq]
    E = np.full((len(lists), 4), -1000, dtype=int)
    for i, l in enumerate(lists):
        E[i, :len(l)] = l[:4]
    idx = pd.Index(uniq).get_indexer(sets)
    ex = E[idx]                                   # (rows, 4), padded with -1000
    valid = ex > -1000
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = valid.sum(axis=1)
    mn = np.where(valid, ex, 1000).min(axis=1)
    mx = np.where(valid, ex, -1000).max(axis=1)
    inset = ((ex == t[:, None]) & valid).any(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), 1000).min(axis=1)
    g_u = np.array([_m.gcd(*l) if len(l) > 1 else 0 for l in lists])
    samelast_u = np.array([len(set(x % 10 for x in l)) == 1 for l in lists])
    g = g_u[idx]
    samelast = samelast_u[idx]
    outside = (t < mn) | (t > mx)
    inrange = (t > mn) & (t < mx)
    far_out = (mn - t > 10) | (t - mx > 10)

    m = (dist > 15) & ~inset
    if m.sum() < 3:
        return 0.0
    x = n[m].astype(float)
    v = y[m]
    xc = x - x.mean()
    den = (xc ** 2).sum()
    return float((xc * (v - v.mean())).sum() / den) if den > 0 else 0.0
