# name: small_rule_member_contrast
# description: For sets of 2+ examples all fitting a small rule (<=12 members: squares, cubes, powers of k, multiples of 8-12, ends-in-digit), yes rate for unshown targets in the smallest such rule minus yes rate for targets outside it; observed < null means the model's without-replacement size principle overcommits to small rules (people generalize less sharply along the rule), observed > null means people are even more rule-bound than the model.
_CACHE = {}
def _rules():
    if "R" in _CACHE:
        return _CACHE["R"]
    rs = [[k * k for k in range(1, 11)], [k ** 3 for k in range(1, 5)]]
    for m in range(2, 11):
        rs.append([m ** k for k in range(1, 8) if m ** k <= 100])
    for m in [8, 9, 10, 11, 12]:
        rs.append(list(range(m, 101, m)))
    for dgt in range(10):
        rs.append([v for v in range(1, 101) if v % 10 == dgt])
    M = np.zeros((len(rs), 101), dtype=bool)
    for i, rr in enumerate(rs):
        M[i, rr] = True
    sizes = M.sum(axis=1)
    _CACHE["R"] = (M, sizes)
    return M, sizes
def test_statistic(df):
    M, sizes = _rules()
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    memb = {}
    for s in uniq:
        if s in _CACHE:
            memb[s] = _CACHE[s]
            continue
        xs = [int(v) for v in s.split(",")]
        row = None
        if len(xs) >= 2:
            ok = M[:, xs].all(axis=1)
            if ok.any():
                idx = np.where(ok)[0][np.argmin(sizes[ok])]
                row = M[idx]
        _CACHE[s] = row
        memb[s] = row
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    inset = (ex == t[:, None]).any(axis=1)
    has = np.array([memb[s] is not None for s in sets])
    inrule = np.zeros(len(df), dtype=bool)
    sv = sets.to_numpy()
    for s, row in memb.items():
        if row is None:
            continue
        m = sv == s
        inrule[m] = row[t[m]]
    a = has & ~inset & inrule
    b = has & ~inset & ~inrule
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
