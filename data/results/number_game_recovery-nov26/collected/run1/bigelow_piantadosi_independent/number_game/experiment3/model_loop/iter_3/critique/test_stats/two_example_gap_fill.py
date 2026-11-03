# name: two_example_gap_fill
# description: In 2-example sets whose examples are 4-30 apart, among non-example targets sharing no final digit with an example and within half the gap of the nearest example, yes-rate for targets strictly between the examples minus those outside them; observed above the null means people interpolate between two close examples more (vs extrapolating) than the model's interval posterior, below means less.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    share_fd = np.any(np.mod(E, 10) == np.mod(t, 10)[:, None], axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    lo = np.nanmin(E, axis=1)
    hi = np.nanmax(E, axis=1)
    gap = hi - lo
    m = (~is_ex) & (n == 2) & (gap >= 4) & (gap <= 30) & (~share_fd) & (dist <= gap / 2.0)
    inside = (t > lo) & (t < hi)
    return rate(m & inside) - rate(m & (~inside))
