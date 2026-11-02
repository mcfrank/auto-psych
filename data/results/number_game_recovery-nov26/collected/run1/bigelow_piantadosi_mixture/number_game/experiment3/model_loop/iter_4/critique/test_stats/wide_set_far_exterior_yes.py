# name: wide_set_far_exterior_yes
# description: For sets of >=2 examples spanning >=40, yes rate for non-example targets more than 15 beyond the span; observed > null means people extrapolate widely spread examples much further than the model's Gaussian reach allows, observed < null means the model over-extends them.
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
    return rate((k >= 2) & (span >= 40) & (beyond > 15) & ~is_ex)
