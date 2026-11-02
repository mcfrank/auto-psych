# name: mixed_parity_even_target_bias
# description: In sets whose examples include both odd and even numbers, yes-rate for even non-example targets minus odd non-example targets; observed above the null means people favour even targets beyond what the model's (parity-symmetric given mixed examples) rules and intervals predict, below means the model over-produces an even bias.
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
    par = np.where(np.isnan(E), np.nan, np.mod(E, 2))
    mixed = (np.nanmax(par, axis=1) == 1) & (np.nanmin(par, axis=1) == 0)
    m = (~is_ex) & mixed
    return rate(m & (np.mod(t, 2) == 0)) - rate(m & (np.mod(t, 2) == 1))
