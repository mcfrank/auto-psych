# name: math_rule_member_contrast
# description: For sets whose smallest consistent rule is an arithmetic rule (squares, cubes, primes, multiples of k, powers of k; size <= 35), yes rate of non-example targets in that rule minus non-example targets outside it; observed below null means people discriminate members of arithmetic concepts less than the model predicts (above null means more).
def test_statistic(df):
    N = np.arange(1, 101)
    rules = []
    rules.append(np.isin(N, [n * n for n in range(1, 11)]))
    rules.append(np.isin(N, [n ** 3 for n in range(1, 5)]))
    rules.append(np.array([n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1)) for n in N]))
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        rules.append(N % m == 0)
    for m in range(2, 11):
        rules.append(np.isin(N, [m ** k for k in range(1, 8) if m ** k <= 100]))
    M = np.array(rules)  # (R,100)
    sizes = M.sum(axis=1)
    # digit rules also compete for "smallest"
    D = np.array([(N % 10) == d for d in range(10)] + [np.array([str(d) in str(n) for n in N]) for d in range(10)])
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    mem_tab = {}
    for s in uniq:
        e = np.array([int(x) for x in s.split(",")]) - 1
        ok = M[:, e].all(axis=1)
        okd = D[:, e].all(axis=1)
        best_d = D.sum(axis=1)[okd].min() if okd.any() else 999
        if ok.any():
            i = np.where(ok)[0][np.argmin(sizes[ok])]
            if sizes[i] <= 35 and sizes[i] < best_d:
                mem_tab[s] = M[i]
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    s_arr = sets.to_numpy()
    valid = np.array([x in mem_tab for x in s_arr])
    if not valid.any():
        return 0.0
    mem = np.zeros(len(df), dtype=bool)
    for s, row in mem_tab.items():
        idx = s_arr == s
        mem[idx] = row[t[idx] - 1]
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    inset = ex.eq(df["target"].astype(int), axis=0).any(axis=1).to_numpy()
    a = valid & ~inset & mem; b = valid & ~inset & ~mem
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
