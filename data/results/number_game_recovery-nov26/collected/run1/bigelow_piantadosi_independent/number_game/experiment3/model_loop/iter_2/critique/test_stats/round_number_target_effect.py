# name: round_number_target_effect
# description: Among sets with no example that is a multiple of 10, yes-rate for non-example targets more than 10 from every example that are multiples of 10 minus other such targets; observed below the null means people avoid round-number targets more than the model predicts (above means they favour them more).
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
    any10 = np.any(fd == 0, axis=1)
    m = (~is_ex) & (~any10) & (dist > 10)
    return rate(m & (np.mod(t, 10) == 0)) - rate(m & (np.mod(t, 10) != 0))
