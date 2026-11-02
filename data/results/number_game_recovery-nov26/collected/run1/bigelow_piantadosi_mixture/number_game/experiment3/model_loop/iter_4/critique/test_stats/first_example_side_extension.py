# name: first_example_side_extension
# description: For sets of >=2 examples whose first example is the min or the max, yes rate for targets 1-20 beyond the span on the first example's side minus on the opposite side; observed > null means people anchor extrapolation on the first example more than the model's sticky first guess predicts, observed < null they favour the later examples' side.
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
    f = ex[:, 0]
    first_lo = (f == lo) & (k >= 2) & (span > 0); first_hi = (f == hi) & (k >= 2) & (span > 0)
    near = (beyond >= 1) & (beyond <= 20)
    below = t < lo; above = t > hi
    same = near & ((first_lo & below) | (first_hi & above))
    other = near & ((first_lo & above) | (first_hi & below))
    return rate(same) - rate(other)
