# name: span_scaled_exterior_reach
# description: Among sets of >=2 examples, yes rate for targets 1-15 beyond the span in sets spanning >=40 minus in sets spanning <=15; observed > null means people scale extrapolation reach with how spread the examples are more than the model (fixed absolute reach), observed < null less.
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
    m = (k >= 2) & (beyond >= 1) & (beyond <= 15)
    return rate(m & (span >= 40)) - rate(m & (span <= 15))
