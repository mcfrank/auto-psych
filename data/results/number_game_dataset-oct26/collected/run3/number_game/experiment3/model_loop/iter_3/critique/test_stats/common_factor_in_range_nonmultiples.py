# name: common_factor_in_range_nonmultiples
# description: For 2+ example sets whose examples share a common factor g >= 2 (excluding sets whose examples all share a last digit), yes-rate for non-example targets strictly inside the [min,max] range that are NOT multiples of g; observed < null means people stick to the factor rule and reject in-range non-multiples more than the model (whose intervals fill the range) predicts (observed > null: people fill the range more).
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

    m = (n >= 2) & (g >= 2) & ~samelast & inrange & ~inset & (t % np.maximum(g, 1) != 0)
    return float(y[m].mean()) if m.any() else 0.0
