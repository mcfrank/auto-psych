# name: above_vs_below_near
# description: Among non-example targets 1-10 from the nearest example and outside the examples' range, yes rate for targets above the max example minus targets below the min example; observed above null means people still extend upward more than the compressed-number-line model predicts, below null means the model now over-produces upward extension.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    lo = np.where(valid, ex, np.inf).min(axis=1)
    hi = np.where(valid, ex, -np.inf).max(axis=1)
    base = nonmember & (nearest <= 10)
    a = r[base & (t > hi)]; b = r[base & (t < lo)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
