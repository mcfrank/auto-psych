# name: incoherent_wide_set_inrange_yes
# description: For sets of 3-4 examples spanning >=50 that fit no small mathematical rule (<=20 members), the yes rate for targets inside [min,max] of the examples; observed below null means people refuse to generalize from rule-less scattered sets more than the model's broad-interval posterior predicts, above means they accept the wide range more.
def test_statistic(df):
    N = np.arange(1, 101)
    pr = [n for n in N if n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))]
    rules = [pr, [n * n for n in range(1, 11)], [n ** 3 for n in range(1, 5)]]
    rules += [list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    rules += [[m ** k for k in range(1, 8) if m ** k <= 100] for m in range(2, 11)]
    rules += [[n for n in N if n % 10 == d] for d in range(10)]
    rules += [[n for n in N if str(d) in str(n)] for d in range(10)]
    M = np.array([np.isin(N, r) for r in rules])
    sizes = M.sum(1)
    sets = df["number_set"].astype(str)
    best = {}
    for s in sets.unique():
        ex = np.array([int(x) for x in s.split(",")])
        ok = M[:, ex - 1].all(1) & (sizes <= 20)
        best[s] = int(np.where(ok)[0][np.argmin(sizes[ok])]) if (ok.any() and len(ex) >= 2) else -1
    ridx = sets.map(best).to_numpy()
    t = df["target"].astype(int).to_numpy()
    has = ridx >= 0
    inrule = np.zeros(len(df), bool)
    inrule[has] = M[ridx[has], t[has] - 1]
    y = df["rating"].astype(float).to_numpy()
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    m = ~has & (n >= 3) & (hi - lo >= 50) & (t >= lo) & (t <= hi)
    return float(y[m].mean()) if m.any() else 0.0
