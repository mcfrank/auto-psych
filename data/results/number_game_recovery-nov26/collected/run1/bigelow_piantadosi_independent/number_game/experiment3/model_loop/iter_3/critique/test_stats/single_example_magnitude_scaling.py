# name: single_example_magnitude_scaling
# description: For single-example sets, yes-rate for non-example targets within 10 of the example when the example is >=50 minus when it is <=20; observed above the null means people's neighbourhood around a large example is wider (Weber-like) than the model's compressed number line implies, below means the model over-produces magnitude scaling.
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
    x = np.nan_to_num(E[:, 0])
    m = (~is_ex) & (n == 1) & (dist <= 10)
    return rate(m & (x >= 50)) - rate(m & (x <= 20))
