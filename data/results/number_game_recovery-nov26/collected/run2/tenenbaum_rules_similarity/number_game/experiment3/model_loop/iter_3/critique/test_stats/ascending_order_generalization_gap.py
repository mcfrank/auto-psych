# name: ascending_order_generalization_gap
# description: For sets of 3+ examples, yes rate on non-example targets when the examples were shown in ascending order minus when shown in non-ascending order; nonzero observed vs null means presentation order changes generalisation (positive = ordered sequences invite broader, rule-like generalisation), which the order-blind model can only produce through set content.
def test_statistic(df):
    s = df["number_set"].astype(str)
    lists = s.str.split(",")
    n = lists.str.len().to_numpy()
    asc_map = {k: all(int(a) < int(b) for a, b in zip(k.split(",")[:-1], k.split(",")[1:])) for k in s.unique()}
    asc = s.map(asc_map).to_numpy().astype(bool)
    t = df["target"].astype(int).astype(str)
    isex = np.array([ti in e for ti, e in zip(t, lists)])
    r = df["rating"].to_numpy()
    a = (n >= 3) & asc & ~isex
    b = (n >= 3) & ~asc & ~isex
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
