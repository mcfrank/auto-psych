# name: rule_discrimination_four_examples
# description: For 4-example sets whose examples all fit a small rule (multiples of m=3..12, squares, powers of 2, or a shared final digit; smallest such rule), yes-rate on out-of-span rule-member targets minus out-of-span non-member targets; observed above null_mean means people follow the rule beyond the span more sharply than the model at n=4 (below = the model over-commits to the rule).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    rules = [set(range(m, 101, m)) for m in range(3, 13)]
    rules.append({k * k for k in range(1, 11)})
    rules.append({2 ** k for k in range(0, 7)})
    rules += [{x for x in range(1, 101) if x % 10 == d} for d in range(10)]
    memb = {}
    for s in sets.unique():
        a = [int(x) for x in s.split(",")]
        best = None
        if len(a) == 4:
            for R in rules:
                if all(x in R for x in a) and (best is None or len(R) < len(best)):
                    best = R
        memb[s] = np.array([(k in best) if best is not None else False for k in range(101)])
        memb[s] = (memb[s], best is not None)
    has = sets.map(lambda s: memb[s][1]).to_numpy().astype(bool)
    t = df["target"].astype(int).to_numpy()
    M = np.stack([memb[s][0] for s in sets.unique()])
    idx = pd.Index(sets.unique()).get_indexer(sets)
    mem = M[idx, t]
    ex = sets.str.split(",", expand=True).astype(float).to_numpy()
    out = (t < np.nanmin(ex, axis=1)) | (t > np.nanmax(ex, axis=1))
    y = df["rating"].astype(float).to_numpy()
    a = has & out & mem
    b = has & out & ~mem
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
