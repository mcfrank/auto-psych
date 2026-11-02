# name: interior_proximity_gradient
# description: Among sets of >=2 examples, for non-example targets strictly inside [min,max] example, yes rate when the nearest example is 1-3 away minus when it is >=12 away; observed > null means people still discount interior targets far from any example, which the model (no hesitancy inside the span) under-produces.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1); span = hi - lo
    dnear = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    is_ex = dnear == 0
    beyond = np.maximum(0, np.maximum(lo - t, t - hi))
    interior = (t > lo) & (t < hi) & ~is_ex
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    m = interior & (k >= 2)
    return rate(m & (dnear <= 3)) - rate(m & (dnear >= 12))
