# name: largest_gap_interior_contrast
# description: For 3-4 example sets consistent with no rule, among non-example targets inside the examples' range, yes rate inside the largest gap between adjacent examples minus inside the other gaps; observed below null_mean means people treat the examples as clusters and reject big gaps more than the model's interval hypotheses predict, above means they fill gaps more uniformly.

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


def _gap(ex):
    M, par = _rule_matrix()
    if len(ex) < 3 or M[:, ex].all(axis=1).any():
        return (0, 0)
    s = np.sort(ex)
    g = np.argmax(np.diff(s))
    return (int(s[g]), int(s[g + 1]))


def test_statistic(df):
    codes, feats = _row_set_features(df, "gap", _gap)
    f = np.array(feats)[codes]
    a, b = f[:, 0], f[:, 1]
    t = df["target"].to_numpy().astype(float)
    dmin, lo, hi, _ = _min_dist(df)
    has = b > a
    inside = has & (dmin > 0) & (t > lo) & (t < hi)
    in_big = inside & (t > a) & (t < b)
    return _diff(df, in_big, inside & ~in_big)
