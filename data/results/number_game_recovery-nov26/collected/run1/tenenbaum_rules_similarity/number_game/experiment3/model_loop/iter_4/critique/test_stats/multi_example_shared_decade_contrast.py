# name: multi_example_shared_decade_contrast
# description: In multi-example sets, yes rate for non-example targets sharing a tens digit (decade, 1-9=0s, 100 excluded) with any example minus yes rate for other non-example targets; observed above null means people use decade membership more than the model's ranges and rules imply, below null means less.
def test_statistic(df):
    s = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    cache = {}
    for x in set(s):
        v = [int(a) for a in x.split(",")]
        cache[x] = (set(v), set(a // 10 for a in v if a < 100), len(v) > 1)
    ex = np.array([ti in cache[x][0] for ti, x in zip(t, s)])
    dec = np.array([(ti < 100) and (ti // 10 in cache[x][1]) for ti, x in zip(t, s)])
    multi = np.array([cache[x][2] for x in s])
    a = y[multi & dec & ~ex]
    b = y[multi & ~dec & ~ex]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
