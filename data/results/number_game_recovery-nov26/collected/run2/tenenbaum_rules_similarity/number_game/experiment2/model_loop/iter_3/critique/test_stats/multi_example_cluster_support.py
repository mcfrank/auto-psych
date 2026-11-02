# name: multi_example_cluster_support
# description: Among 3-4-example sets and non-example targets, yes rate for targets within 8 of two or more examples minus targets within 8 of exactly one example; observed above the null means similarity support sums across nearby examples more than the model's interval/rule posterior implies (below = less density sensitivity than the model).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ex = {s: np.array([int(x) for x in s.split(",")]) for s in sets.unique()}
    t = df["target"].astype(int).to_numpy()
    n = np.array([len(ex[s]) for s in sets])
    dmin = np.array([np.abs(ex[s] - tt).min() for s, tt in zip(sets, t)])
    cnt = np.array([int((np.abs(ex[s] - tt) <= 8).sum()) for s, tt in zip(sets, t)])
    ok = (n >= 3) & (dmin > 0)
    a, b = ok & (cnt >= 2), ok & (cnt == 1)
    y = df["rating"].to_numpy().astype(float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
