# name: inside_minus_outside_span_multi
# description: For sets of >=3 examples, yes-rate on non-example targets strictly inside [min,max] of the examples minus yes-rate on targets outside that span; observed > null means people fill in the examples' span (interval-like generalization) more strongly relative to extrapolation than the model predicts.
def test_statistic(df):
    sp = df["number_set"].str.split(",")
    n = sp.str.len().to_numpy()
    mn = df["number_set"].map(lambda s: min(int(x) for x in s.split(","))).to_numpy()
    mx = df["number_set"].map(lambda s: max(int(x) for x in s.split(","))).to_numpy()
    exs = df["number_set"].map(lambda s: set(int(x) for x in s.split(","))).to_numpy()
    t = df["target"].astype(int).to_numpy()
    isex = np.array([tt in e for tt, e in zip(t, exs)])
    r = df["rating"].to_numpy().astype(float)
    base = (n >= 3) & ~isex
    a = base & (t > mn) & (t < mx)
    b = base & ((t < mn) | (t > mx))
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
