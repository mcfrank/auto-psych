# name: square_set_square_target_rate
# description: Among sets of >=2 examples that are all perfect squares, yes rate to non-example square targets minus yes rate to non-square targets; observed above the null means people pick up "square numbers" more sharply than the model predicts, below means the model over-commits to squares.
def test_statistic(df):
    sq = {i * i for i in range(1, 11)}
    ex = df["number_set"].astype(str).str.split(",")
    vals = ex.map(lambda e: [int(x) for x in e])
    n = vals.map(len).to_numpy()
    allsq = vals.map(lambda v: all(x in sq for x in v)).to_numpy()
    t = df["target"].astype(int).to_numpy()
    isex = np.array([a in v for a, v in zip(t, vals)])
    r = df["rating"].to_numpy()
    issq = np.isin(t, list(sq))
    base = (n >= 2) & allsq & ~isex
    ms, mn = base & issq, base & ~issq
    if ms.sum() == 0 or mn.sum() == 0:
        return 0.0
    return float(r[ms].mean() - r[mn].mean())
