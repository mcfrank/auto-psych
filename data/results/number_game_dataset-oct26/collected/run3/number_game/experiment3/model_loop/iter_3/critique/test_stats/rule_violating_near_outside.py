# name: rule_violating_near_outside
# description: For 2+ example sets with an arithmetic regularity (common factor >= 2 or a shared last digit), yes-rate for targets 1-5 outside the [min,max] range that violate it (not a multiple of the factor, and not sharing the last digit); observed > null means people extend by numeric proximity across the rule more than the model predicts (observed < null: people respect the rule at the range edge more strictly than the model, whose intervals spill over).
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

    lastd = np.where(valid, ex % 10, -1)[:, 0]
    rule = (n >= 2) & ((g >= 2) | samelast)
    viol_g = (g < 2) | (t % np.maximum(g, 1) != 0)
    viol_l = ~samelast | (t % 10 != lastd)
    near = ((mn - t >= 1) & (mn - t <= 5)) | ((t - mx >= 1) & (t - mx <= 5))
    m = rule & viol_g & viol_l & near & ~inset
    return float(y[m].mean()) if m.any() else 0.0
