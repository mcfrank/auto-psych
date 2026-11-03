# name: first_minus_last_example_proximity
# description: For sets of 2+ examples, yes rate for non-example targets within 5 of the first-shown example (and nearer it than the last) minus those within 5 of the last-shown example (nearer it than the first); observed > 0 beyond model means presentation order (first-example anchoring) matters, which the order-invariant model ignores.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    mn = np.nanmin(ex, axis=1)
    mx = np.nanmax(ex, axis=1)
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    last = ex[np.arange(len(ex)), n - 1]
    d1 = np.abs(ex[:, 0] - t)
    dl = np.abs(last - t)
    m = (n >= 2) & (dist > 0)
    return rate(m & (d1 <= 5) & (d1 < dl)) - rate(m & (dl <= 5) & (dl < d1))
