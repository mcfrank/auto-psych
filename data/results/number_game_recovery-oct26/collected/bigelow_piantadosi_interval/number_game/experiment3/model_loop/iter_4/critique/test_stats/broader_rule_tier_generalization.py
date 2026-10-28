# name: broader_rule_tier_generalization
# description: In multi-example sets fitting a non-parity rule, yes rate on non-example targets that belong to a broader consistent rule but not the smallest one (e.g. 85 or 82 for 80,70,90,100) minus targets in no consistent rule; observed above null_mean means people generalize to superordinate rules more than the model's size-principle posterior predicts, below means less.

_CACHE = {}


def _rule_matrix():
    if "M" in _CACHE:
        return _CACHE["M"], _CACHE["par"]
    N = list(range(1, 101))
    R = [[2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
         [7, 49], [8, 64], [9, 81], [10, 100]]
    R += [[i for i in N if i % n == 0] for n in range(3, 13)]
    R += [[i for i in N if i % 10 == d] for d in range(1, 10)]
    R += [[i for i in N if str(d) in str(i)] for d in range(1, 10)]
    R.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    R.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    R.append([1, 8, 27, 64])
    R.append([i for i in N if i % 2 == 0])
    R.append([i for i in N if i % 2 == 1])
    M = np.zeros((len(R), 101), dtype=bool)
    for k, r in enumerate(R):
        M[k, r] = True
    par = np.zeros(len(R), dtype=bool)
    par[-2:] = True  # even / odd
    _CACHE["M"], _CACHE["par"] = M, par
    return M, par


def _parse(s):
    return np.array([int(x) for x in str(s).split(",")], dtype=int)


def _diff(df, mask_a, mask_b):
    y = df["rating"].to_numpy().astype(float)
    a = y[mask_a].mean() if mask_a.sum() > 0 else 0.0
    b = y[mask_b].mean() if mask_b.sum() > 0 else 0.0
    return float(a - b)


def _row_set_features(df, key, fn):
    # fn(examples) -> tuple of per-set features; cached by set string
    store = _CACHE.setdefault(key, {})
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    feats = []
    for s in uniq:
        if s not in store:
            store[s] = fn(_parse(s))
        feats.append(store[s])
    return codes, feats


def _min_dist(df):
    # distance from target to nearest example (vectorised via padded matrix)
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].to_numpy().astype(float)
    d = np.abs(ex - t[:, None])
    return np.nanmin(d, axis=1), np.nanmin(ex, axis=1), np.nanmax(ex, axis=1), ex


def _tiers(ex):
    M, par = _rule_matrix()
    sizes = M.sum(axis=1)
    if len(ex) < 2:
        return None
    cons = M[:, ex].all(axis=1)
    c = np.flatnonzero(cons & (~par) & (sizes <= 34))
    if len(c) == 0:
        return None
    small = c[np.argmin(sizes[c])]
    return (M[small], M[cons].any(axis=0))


def test_statistic(df):
    codes, feats = _row_set_features(df, "tiers", _tiers)
    S = np.zeros((len(feats), 101), dtype=bool)
    U = np.zeros((len(feats), 101), dtype=bool)
    has = np.zeros(len(feats), dtype=bool)
    for i, f in enumerate(feats):
        if f is not None:
            S[i], U[i] = f
            has[i] = True
    t = df["target"].to_numpy().astype(int)
    dmin, _, _, _ = _min_dist(df)
    base = has[codes] & (dmin > 0) & ~S[codes, t]
    broader = base & U[codes, t]
    none = base & ~U[codes, t]
    return _diff(df, broader, none)
