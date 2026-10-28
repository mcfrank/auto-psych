# name: edge_step_sharpness
# description: Among out-of-span non-example targets, yes-rate at gap 1-5 from the nearest span edge minus yes-rate at gap 6-15; observed > null means generalization drops off more sharply just past the span edge than the model's smooth exponential size similarity produces, observed < null means a flatter fall-off.
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
    m = ~is_ex & (gap >= 1)
    return rate(m & (gap <= 5)) - rate(m & (gap >= 6) & (gap <= 15))
