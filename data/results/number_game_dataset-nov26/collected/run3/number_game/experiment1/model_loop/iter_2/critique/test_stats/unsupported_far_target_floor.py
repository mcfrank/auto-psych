# name: unsupported_far_target_floor
# description: Yes rate for non-example targets that lie outside the [min,max] example range, more than 5 away from every example, and belong to no small rule (<=20 members, of the model's 45) containing all examples; observed above null means people have a guessing/lapse floor the model lacks, below null means they reject such targets even more firmly.
def test_statistic(df):
    N = np.arange(1, 101)
    rules = [N % 2 == 1, N % 2 == 0,
             np.array([n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1)) for n in N]),
             np.isin(N, [n * n for n in range(1, 11)]), np.isin(N, [n ** 3 for n in range(1, 5)]),
             np.isin(N, [2, 4, 8, 16, 64]), np.isin(N, [2, 4, 8, 16, 32, 37, 64])]
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        rules.append(N % m == 0)
    for m in range(2, 11):
        rules.append(np.isin(N, [m ** k for k in range(1, 8) if m ** k <= 100]))
    for d in range(10):
        rules.append(N % 10 == d)
        rules.append(np.array([str(d) in str(n) for n in N]))
    M = np.array(rules)
    sets = df["number_set"].astype(str)
    s_arr = sets.to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    sel = np.zeros(len(df), dtype=bool)
    for s in sets.unique():
        e = np.array([int(x) for x in s.split(",")])
        ok = M[:, e - 1].all(axis=1) & (M.sum(axis=1) <= 20)
        cover = M[ok].any(axis=0) if ok.any() else np.zeros(100, dtype=bool)
        far = np.abs(N[:, None] - e[None, :]).min(axis=1) > 5
        good = far & ~cover & ((N < e.min()) | (N > e.max()))
        idx = s_arr == s
        sel[idx] = good[t[idx] - 1]
    return float(r[sel].mean()) if sel.any() else 0.0
