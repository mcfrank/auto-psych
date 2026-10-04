# name: balance_pref_equal_switches
# description: Among pairs with equal switch counts but different H/T imbalance |#H-#T|, rate of choosing the more balanced sequence; observed above null_mean means people use H/T balance, which the alternation-only model under-uses.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    u = pd.unique(pd.concat([a, b]))
    sw = {s: sum(1 for x, y in zip(s, s[1:]) if x != y) for s in u}
    im = {s: abs(s.count("H") - s.count("T")) for s in u}
    ia = a.map(im).to_numpy(); ib = b.map(im).to_numpy()
    m = (a.map(sw).to_numpy() == b.map(sw).to_numpy()) & (ia != ib)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy(dtype=float)[m]
    return float(np.mean(np.where((ia < ib)[m], y, 1 - y)))
