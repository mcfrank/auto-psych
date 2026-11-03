# name: pure_digit_set_neighbour_leak
# description: In 2-4 example sets whose examples all share one final digit, yes-rate for non-example targets within 2 of an example that do not share that final digit; observed above the null means people leak magnitude similarity around a clean digit-rule set more than the model (which concentrates on the rule), below means the model leaks too much.
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
    fdm = np.where(np.isnan(E), np.nan, fd)
    pure = (np.nanmax(fdm, axis=1) == np.nanmin(fdm, axis=1)) & (n >= 2)
    m = (~is_ex) & pure & (dist <= 2) & (~share_fd)
    return rate(m)
