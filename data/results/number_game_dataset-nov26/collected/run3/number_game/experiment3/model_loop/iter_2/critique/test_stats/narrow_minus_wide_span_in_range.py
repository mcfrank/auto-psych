# name: narrow_minus_wide_span_in_range
# description: On 2+ example sets, yes rate for in-range non-example targets when the examples span at most 30 minus when they span more than 60; observed below null means the model over-endorses the interior of narrow example ranges (tight intervals) or under-endorses the interior of wide ones relative to people.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    U = E % 10; tu = t % 10
    def rate(m):
        return float(y[m].mean()) if m.any() else 0.0
    span = hi - lo
    base = (n >= 2) & ~is_ex & (t >= lo) & (t <= hi)
    return rate(base & (span <= 30)) - rate(base & (span > 60))
