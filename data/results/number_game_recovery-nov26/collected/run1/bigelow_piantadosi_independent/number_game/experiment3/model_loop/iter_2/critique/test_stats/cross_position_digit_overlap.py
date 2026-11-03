# name: cross_position_digit_overlap
# description: In 2-4 example sets, among non-example targets more than 10 from every example and sharing no final digit with any example, yes-rate for targets whose final digit equals some example tens digit or whose tens digit equals some example final digit, minus the rest; observed above the null means people use loose cross-position digit matching that the model's whole-set digit rules cannot produce.
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
    tens = np.floor(E / 10)
    cross = np.any((tens == np.mod(t, 10)[:, None]) | (fd == np.floor(t / 10)[:, None]), axis=1)
    m = (~is_ex) & (~share_fd) & (dist > 10) & (n >= 2)
    return rate(m & cross) - rate(m & ~cross)
