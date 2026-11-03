# name: inside_range_far
# description: Among non-example targets more than 5 from the nearest example, yes rate for targets inside the examples' [min,max] range minus targets outside it; observed above null means people fill in the spanned range more than the compressed-interval model predicts, below null means the model over-produces in-range filling.
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
    m = nonmember & (nearest > 5)
    ins = (t >= lo) & (t <= hi)
    a = r[m & ins]; b = r[m & ~ins]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
