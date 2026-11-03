# name: single_example_near_gradient
# description: For single-example sets, yes-rate for non-example targets 1-3 from the example minus targets 4-10 from it; observed above the null means people's generalization from a lone example falls off within the first few numbers faster than the fitted width prior implies, below means the model's near falloff is too steep.
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
    m = (~is_ex) & (n == 1)
    return rate(m & (dist <= 3)) - rate(m & (dist >= 4) & (dist <= 10))
