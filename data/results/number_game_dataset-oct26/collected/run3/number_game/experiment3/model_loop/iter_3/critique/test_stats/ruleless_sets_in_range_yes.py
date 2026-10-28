# name: ruleless_sets_in_range_yes
# description: For 3-4 example sets with no common factor, no shared last digit and mixed parity (sets the model can only explain with an interval), yes-rate for non-example targets strictly inside the [min,max] range; observed < null means people do not fill the gaps of patternless sets as an interval learner would, treating them as random (observed > null: they fill the range more than the model).
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

    par = np.where(valid, ex % 2, -1)
    mixed = ((par == 0).any(axis=1)) & ((par == 1).any(axis=1))
    m = (n >= 3) & (g == 1) & ~samelast & mixed & inrange & ~inset
    return float(y[m].mean()) if m.any() else 0.0
