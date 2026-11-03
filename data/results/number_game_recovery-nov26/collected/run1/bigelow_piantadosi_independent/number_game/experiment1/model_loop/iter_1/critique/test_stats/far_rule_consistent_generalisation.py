# name: far_rule_consistent_generalisation
# description: Yes rate among non-example targets more than 10 from every example that are divisible by the examples' common divisor (gcd >= 2) — i.e. distant targets licensed only by an arithmetic rule; observed above null means people generalise broad arithmetic rules to distant numbers more than the model predicts, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)  # (rows, <=4), NaN pad
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    dist = np.abs(ex - t[:, None])
    nearest = np.nanmin(dist, axis=1)
    nonmember = nearest > 0
    e = np.nan_to_num(ex, nan=0.0).astype(np.int64)
    g = np.gcd.reduce(e, axis=1)
    m = nonmember & (nearest > 10) & (g >= 2) & (np.mod(t.astype(np.int64), np.maximum(g, 1)) == 0)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
