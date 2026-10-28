# name: odd_set_parity_contrast
# description: Among sets of >=2 all-odd examples that do NOT share a ones digit, yes rate to odd non-example targets minus yes rate to even targets; observed above the null means people use parity ("odd numbers") more sharply than the model predicts, below means the model over-weights parity.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",")
    vals = ex.map(lambda e: [int(x) for x in e])
    n = vals.map(len).to_numpy()
    allodd = vals.map(lambda v: all(x % 2 == 1 for x in v)).to_numpy()
    shared = vals.map(lambda v: len({x % 10 for x in v}) == 1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    isex = np.array([a in v for a, v in zip(t, vals)])
    r = df["rating"].to_numpy()
    base = (n >= 2) & allodd & ~shared & ~isex
    mo, me = base & (t % 2 == 1), base & (t % 2 == 0)
    if mo.sum() == 0 or me.sum() == 0:
        return 0.0
    return float(r[mo].mean() - r[me].mean())
