# name: small_rule_member_contrast
# description: For sets of 2+ examples that all fall in some listed rule with at most 12 members (powers of 2/3/4, n^2, squares, cubes, multiples of 10, etc.; smallest such rule used), yes rate on non-example rule-member targets minus non-member targets; observed below null_mean means people generalize less sharply along small mathematical rules than the model (model too all-or-none), above means sharper.
def test_statistic(df):
    N = range(1, 101)
    rules = [[2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36], [7, 49], [8, 64], [9, 81], [10, 100],
             [1, 4, 9, 16, 25, 36, 49, 64, 81, 100], [1, 8, 27, 64]]
    rules += [[i for i in N if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in N if i % 10 == d] for d in range(1, 10)]
    rules = [set(x) for x in rules if len(x) <= 12]
    best = {}
    for s in df["number_set"].unique():
        e = set(int(x) for x in str(s).split(","))
        if len(e) < 2:
            continue
        c = [x for x in rules if e <= x]
        if c:
            best[s] = min(c, key=len)
    if not best:
        return 0.0
    sub = df[df["number_set"].isin(list(best))]
    t = sub["target"].astype(int)
    mem = pd.Series([tt in best[s] for s, tt in zip(sub["number_set"], t)], index=sub.index)
    isex = pd.Series([str(tt) in str(s).split(",") for s, tt in zip(sub["number_set"], t)], index=sub.index)
    r = sub["rating"].astype(float)
    a = mem & ~isex
    b = ~mem
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
