# name: odd_sets_far_parity_contrast
# description: For 2+ example sets whose examples are all odd (excluding sets that all share a last digit), among targets more than 10 outside the [min,max] range, yes-rate for odd targets minus even targets; observed > null means people generalize an odd-number concept to far targets more sharply than the model (observed < null: less parity-selective than the model).
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

    allodd = ((ex % 2 == 1) | ~valid).all(axis=1)
    m = (n >= 2) & allodd & ~samelast & far_out & ~inset
    o = m & (t % 2 == 1)
    e = m & (t % 2 == 0)
    if not o.any() or not e.any():
        return 0.0
    return float(y[o].mean() - y[e].mean())
