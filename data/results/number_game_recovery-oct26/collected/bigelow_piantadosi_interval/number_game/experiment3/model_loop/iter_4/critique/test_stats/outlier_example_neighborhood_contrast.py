# name: outlier_example_neighborhood_contrast
# description: In 4-example sets where 3 examples fit a non-parity rule and one is an outlier, among non-example non-rule targets, yes rate within 5 of the outlier example minus yes rate within 5 of a rule-fitting example; observed below null_mean means people discount the outlier's neighbourhood more than the model (which builds ranges equally around all examples), above means they treat the outlier as more central.

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


def _majority(ex):
    M, par = _rule_matrix()
    if len(ex) != 4:
        return (-1, -1)
    cnt = M[:, ex].sum(axis=1)
    sizes = M.sum(axis=1)
    ok = (~par) & (sizes <= 34)
    if np.any(ok & (cnt == 4)):
        return (-1, -1)
    cand = np.flatnonzero(ok & (cnt == 3))
    if len(cand) == 0:
        return (-1, -1)
    r = int(cand[np.argmin(sizes[cand])])
    outlier = int(ex[~M[r, ex]][0])
    return (r, outlier)


def test_statistic(df):
    M, par = _rule_matrix()
    codes, feats = _row_set_features(df, "majout", _majority)
    f = np.array(feats)[codes]
    rule, outl = f[:, 0], f[:, 1]
    t = df["target"].to_numpy().astype(int)
    _, _, _, ex = _min_dist(df)
    is_ex = (ex == t[:, None]).any(axis=1)
    member = M[np.maximum(rule, 0), t]
    d_out = np.abs(t - outl)
    exm = np.where(ex == outl[:, None], np.nan, ex)
    with np.errstate(all="ignore"):
        d_maj = np.nanmin(np.abs(exm - t[:, None]), axis=1)
    valid = (rule >= 0) & ~is_ex & ~member
    near_out = valid & (d_out <= 5) & (d_maj > 5)
    near_maj = valid & (d_maj <= 5) & (d_out > 5)
    return _diff(df, near_out, near_maj)
