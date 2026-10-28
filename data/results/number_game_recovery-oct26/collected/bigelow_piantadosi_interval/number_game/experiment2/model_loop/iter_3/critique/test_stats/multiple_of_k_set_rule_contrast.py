# name: multiple_of_k_set_rule_contrast
# description: Among sets of >=2 examples whose gcd k is 3..12 and that do not share a ones digit, yes rate to non-example multiples of k minus yes rate to non-multiples; observed above the null means people commit to the "multiples of k" rule more sharply than the model predicts, below means the model over-commits to it.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    vals = ex.map(lambda e: [int(x) for x in e])
    n = vals.map(len).to_numpy()
    g = vals.map(lambda v: math.gcd(*v) if len(v) > 1 else 0).to_numpy()
    shared = vals.map(lambda v: len({x % 10 for x in v}) == 1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    isex = np.array([a in v for a, v in zip(t, vals)])
    r = df["rating"].to_numpy()
    base = (n >= 2) & (g >= 3) & (g <= 12) & ~shared & ~isex
    gs = np.where(g > 0, g, 1)
    mm, mn = base & (t % gs == 0), base & (t % gs != 0)
    if mm.sum() == 0 or mn.sum() == 0:
        return 0.0
    return float(r[mm].mean() - r[mn].mean())
