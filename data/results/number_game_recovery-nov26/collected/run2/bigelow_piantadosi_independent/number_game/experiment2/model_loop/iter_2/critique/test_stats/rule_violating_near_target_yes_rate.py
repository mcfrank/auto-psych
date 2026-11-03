# name: rule_violating_near_target_yes_rate
# description: Among sets of >=2 examples that all fit some small mathematical rule (<=20 members), the yes rate for targets NOT in that rule but within 5 of some example; observed above null means people let numerical proximity override the rule more than the model does (similarity leak), below means people respect the rule more strictly than the model.
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
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = has & ~inrule & (dist <= 5)
    return float(y[m].mean()) if m.any() else 0.0
