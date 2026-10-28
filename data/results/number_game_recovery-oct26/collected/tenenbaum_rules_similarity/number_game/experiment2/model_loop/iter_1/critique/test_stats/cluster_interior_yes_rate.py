# name: cluster_interior_yes_rate
# description: Yes rate to non-example targets strictly inside the span of the examples, on sets of >=2 examples spanning at most 20; observed above null means the model under-produces "yes" inside tight clusters (people treat them as ranges more than the model), below null means it over-produces them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    t = df["target"].astype(float).to_numpy()
    is_ex = np.any(ex == t[:, None], axis=1)
    m = (n >= 2) & (hi - lo <= 20) & (t > lo) & (t < hi) & ~is_ex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
