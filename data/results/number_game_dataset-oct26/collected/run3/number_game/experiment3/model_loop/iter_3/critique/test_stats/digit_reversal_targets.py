# name: digit_reversal_targets
# description: Yes-rate for targets that are the digit reversal of some two-digit example (e.g. 72 for example 27, both digits nonzero and different) and that share no last digit with any example; observed > null means people treat the examples digits, regardless of position, as part of the concept, which the model has no rule for (observed < null: fewer such yeses than the model predicts).
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

    tens = ex // 10
    ones = ex % 10
    rev = ones * 10 + tens
    isrev = (valid & (ex >= 10) & (ex < 100) & (ones != 0) & (ones != tens) & (rev == t[:, None])).any(axis=1)
    sharelast = ((ex % 10 == (t % 10)[:, None]) & valid).any(axis=1)
    m = isrev & ~sharelast & ~inset
    return float(y[m].mean()) if m.any() else 0.0
