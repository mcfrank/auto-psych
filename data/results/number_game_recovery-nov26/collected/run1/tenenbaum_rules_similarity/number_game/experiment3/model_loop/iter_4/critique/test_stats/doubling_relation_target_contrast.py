# name: doubling_relation_target_contrast
# description: Yes rate for non-example targets that are double or half of some example, minus yes rate for other non-example targets of the same sets; observed above null means people generalise along multiplicative (x2) relations more than the model's rule menu predicts, below null means less.
def test_statistic(df):
    s = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    cache = {}
    for x in set(s):
        v = [int(a) for a in x.split(",")]
        rel = set([2 * a for a in v] + [a // 2 for a in v if a % 2 == 0]) - set(v)
        cache[x] = (set(v), rel)
    ex = np.array([ti in cache[x][0] for ti, x in zip(t, s)])
    rel = np.array([ti in cache[x][1] for ti, x in zip(t, s)])
    a = y[rel & ~ex]
    b = y[~rel & ~ex]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
