# name: wide_span_interior_gap_endorsement
# description: In 3-4 example sets spanning at least 60, yes-rate for non-example targets strictly inside the span, more than 8 from every example and sharing no final digit with any example; observed above the null means people fill the interior of widely spread sets more (diffuse range responding) than the model, below means the model's wide intervals over-fill it.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    fd = np.mod(E, 10)
    share_fd = np.any(fd == np.mod(t, 10)[:, None], axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    lo = np.nanmin(E, axis=1)
    hi = np.nanmax(E, axis=1)
    m = (~is_ex) & (n >= 3) & (hi - lo >= 60) & (t > lo) & (t < hi) & (dist > 8) & (~share_fd)
    return rate(m)
