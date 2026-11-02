# name: decade_boundary_near_extrapolation
# description: For 1-2 example sets, among non-example targets outside the span within 6 of the nearest example, yes rate when the target shares that example tens digit minus when it crosses into another decade; observed above null means decade boundaries limit generalization more than the decade-blind model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    dist = x.sub(t, axis=0).abs()
    nd = dist.min(axis=1)
    d2 = dist.fillna(999).values
    near = x.values[np.arange(len(x)), d2.argmin(axis=1)]
    m = ((n <= 2) & ~isin & (nd <= 6) & ((t < mn) | (t > mx))).values
    sd = (t.values // 10) == (near // 10)
    a = r.values[m & sd].mean() if (m & sd).any() else np.nan
    b = r.values[m & ~sd].mean() if (m & ~sd).any() else np.nan
    v = a - b
    return float(0.0 if (v is None or np.isnan(v)) else v)
