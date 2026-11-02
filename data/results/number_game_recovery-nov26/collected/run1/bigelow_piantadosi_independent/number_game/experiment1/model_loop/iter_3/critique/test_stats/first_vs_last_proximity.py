# name: first_vs_last_proximity
# description: For multi-example sets, yes rate among non-example targets within 3 of only the first-shown example minus those within 3 of only the last-shown example; nonzero observed vs null near 0 means presentation order matters (positive = primacy, negative = recency), which the order-invariant model cannot produce.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    last = n - 1
    idx = np.arange(len(t))
    d_first = dist[:, 0]
    d_last = dist[idx, last]
    nearc = (dist <= 3).sum(axis=1)
    m = nonmember & (n >= 2) & (nearc == 1)
    a = r[m & (d_first <= 3)]; b = r[m & (d_last <= 3)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
