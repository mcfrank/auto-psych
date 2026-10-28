# name: far_generalization_by_set_size
# description: Yes-rate on out-of-span non-example targets at least 20 from the nearest span edge for 3-4-example sets minus 1-2-example sets; observed < null means more examples narrow far generalization more than the model produces (size principle under-produced), observed > null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(ex)), k - 1]
    mn = np.nanmin(ex, axis=1)
    mx = np.nanmax(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    gap = np.maximum(mn - t, t - mx)
    far = (gap >= 20) & ~is_ex
    return rate(far & (k >= 3)) - rate(far & (k <= 2))
