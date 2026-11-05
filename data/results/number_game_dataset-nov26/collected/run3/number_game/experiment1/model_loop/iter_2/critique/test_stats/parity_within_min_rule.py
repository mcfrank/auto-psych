# name: parity_within_min_rule
# description: For sets whose examples all share parity, among non-example targets in the smallest consistent non-parity rule (multiples of k, squares, powers, ends-in/contains-digit, etc.) that contains both parities, yes rate for same-parity targets minus opposite-parity targets; observed above null means people form parity conjunctions (e.g. odd multiples of 3) more than the model's union of single rules predicts.
def test_statistic(df):
    N = np.arange(1, 101)
    rules = [np.isin(N, [n * n for n in range(1, 11)]), np.isin(N, [n ** 3 for n in range(1, 5)])]
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        rules.append(N % m == 0)
    for m in range(2, 11):
        rules.append(np.isin(N, [m ** k for k in range(1, 8) if m ** k <= 100]))
    for d in range(10):
        rules.append(np.array([str(d) in str(n) for n in N]))
    M = np.array(rules)
    mixed = (M & (N % 2 == 0)).any(axis=1) & (M & (N % 2 == 1)).any(axis=1)
    sizes = M.sum(axis=1)
    sets = df["number_set"].astype(str)
    s_arr = sets.to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    mem = np.zeros(len(df), dtype=bool)
    same = np.zeros(len(df), dtype=bool)
    valid = np.zeros(len(df), dtype=bool)
    for s in sets.unique():
        e = np.array([int(x) for x in s.split(",")])
        if len(set(e % 2)) != 1:
            continue
        ok = M[:, e - 1].all(axis=1) & mixed
        if not ok.any():
            continue
        i = np.where(ok)[0][np.argmin(sizes[ok])]
        idx = s_arr == s
        valid[idx] = True
        mem[idx] = M[i][t[idx] - 1] & ~np.isin(t[idx], e)
        same[idx] = (t[idx] % 2) == (e[0] % 2)
    a = valid & mem & same; b = valid & mem & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
