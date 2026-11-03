# name: single_example_distance_gradient
# description: For single-example sets, yes-rate for non-example targets within 5 of the example minus targets more than 20 from it; observed above the null means people generalize from one example with a steeper magnitude-similarity gradient than the model's interval posterior, below means the model's gradient is too steep.
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
    m = (~is_ex) & (n == 1)
    return rate(m & (dist <= 5)) - rate(m & (dist > 20))
