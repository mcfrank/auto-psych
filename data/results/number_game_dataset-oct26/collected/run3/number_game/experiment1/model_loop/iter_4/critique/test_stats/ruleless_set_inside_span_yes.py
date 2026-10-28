# name: ruleless_set_inside_span_yes
# description: Yes-rate on non-example targets strictly inside the examples' span, for 2+-example sets that no arithmetic rule (of the model's 42) jointly covers; observed below null means people endorse the gaps of rule-less, interval-only sets less than the interval hypothesis predicts (above: more).
def test_statistic(df):
    N = np.arange(1, 101)
    rules = [[2,4,8,16,32,64],[3,9,27,81],[4,16,64],[5,25],[6,36],[7,49],[8,64],[9,81],[10,100]]
    rules += [list(N[N % k == 0]) for k in range(3, 13)]
    rules += [list(N[N % 10 == d]) for d in range(1, 10)]
    rules += [[i for i in N if str(d) in str(i)] for d in range(1, 10)]
    rules.append([1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97])
    rules += [list(N[N % 2 == 0]), list(N[N % 2 == 1]), [1,4,9,16,25,36,49,64,81,100], [1,8,27,64]]
    M = np.zeros((len(rules), 102), dtype=bool)
    for i, c in enumerate(rules):
        M[i, c] = True
    M[:, 101] = True  # padding slot for missing examples
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = np.sum(~np.isnan(ex), axis=1)
    idx = np.where(np.isnan(ex), 101, ex).astype(int)
    covered = M[:, idx].all(axis=2).any(axis=0)  # (rows,)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    m = (n >= 2) & ~covered & (t > lo) & (t < hi) & ~member
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
