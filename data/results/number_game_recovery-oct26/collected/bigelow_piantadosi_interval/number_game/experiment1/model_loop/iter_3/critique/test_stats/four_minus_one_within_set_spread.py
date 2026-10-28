# name: four_minus_one_within_set_spread
# description: Mean over sets of the SD across targets of per-(set,target) yes rates, for 4-example sets minus 1-example sets; observed > null means people sharpen (differentiate targets) with more examples more than the model, observed < null means less.
def _features(df):
    # Per unique set: examples, n, min, max, smallest consistent rule (membership over 1..100).
    N = np.arange(1, 101)
    r = [[2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36], [7, 49], [8, 64], [9, 81], [10, 100]]
    r += [[i for i in N if i % k == 0] for k in range(3, 13)]
    r += [[i for i in N if i % 10 == d] for d in range(1, 10)]
    r += [[i for i in N if str(d) in str(i)] for d in range(1, 10)]
    r.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    r.append([i for i in N if i % 2 == 0]); r.append([i for i in N if i % 2 == 1])
    r.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100]); r.append([1, 8, 27, 64])
    M = np.array([np.isin(N, x) for x in r])
    sizes = M.sum(axis=1)
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    nset = len(uniq)
    ex_mask = np.zeros((nset, 100), dtype=bool)
    rule_mask = np.zeros((nset, 100), dtype=bool)
    digits = np.zeros((nset, 10), dtype=bool)
    info = np.zeros((nset, 4))  # n, lo, hi, best rule size (0 if none)
    for i, s in enumerate(uniq):
        e = np.array([int(x) for x in s.split(",")])
        ex_mask[i, e - 1] = True
        ok = M[:, e - 1].all(axis=1)
        bs = 0
        if ok.any():
            j = np.where(ok)[0][np.argmin(sizes[ok])]
            rule_mask[i] = M[j]
            bs = sizes[j]
        for ch in "".join(str(x) for x in e):
            digits[i, int(ch)] = True
        info[i] = [len(e), e.min(), e.max(), bs]
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    f = {
        "t": t, "y": y, "code": codes,
        "n": info[codes, 0], "lo": info[codes, 1], "hi": info[codes, 2], "bs": info[codes, 3],
        "is_ex": ex_mask[codes, t - 1], "in_rule": rule_mask[codes, t - 1],
    }
    td = np.zeros((100, 10), dtype=bool)
    for v in range(1, 101):
        for ch in str(v):
            td[v - 1, int(ch)] = True
    f["share_digit"] = (td[t - 1] & digits[codes]).any(axis=1)
    exs = np.where(ex_mask)
    # min distance from target to any example
    dist = np.abs(np.arange(1, 101)[None, :] - (np.arange(1, 101)[:, None]))  # (example, target)
    mind = np.full((nset, 100), 1000)
    for i in range(nset):
        mind[i] = dist[ex_mask[i]].min(axis=0)
    f["mind"] = mind[codes, t - 1]
    return f


def _rate(y, m):
    return float(y[m].mean()) if m.sum() > 0 else 0.0


def test_statistic(df):
    f = _features(df)
    d = pd.DataFrame({"c": f["code"], "t": f["t"], "y": f["y"], "n": f["n"]})
    cell = d.groupby(["c", "t"]).agg(y=("y", "mean"), n=("n", "first")).reset_index()
    sd = cell.groupby("c").agg(s=("y", "std"), n=("n", "first")).dropna()
    a = sd.loc[sd["n"] == 4, "s"]; b = sd.loc[sd["n"] == 1, "s"]
    return float((a.mean() if len(a) else 0.0) - (b.mean() if len(b) else 0.0))
