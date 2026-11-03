# name: tight_cluster_outside_near
# description: For sets of 2+ examples spanning at most 15, yes rate on targets 1-5 outside the span (below min or above max); observed above null means people stretch a tight cluster's range further than the model predicts, below means they stop at the examples' edges.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    nd = x.sub(t, axis=0).abs().min(axis=1)
    out = ((t > mx) & (t - mx <= 5)) | ((t < mn) & (mn - t <= 5))
    sel = (n >= 2) & ((mx - mn) <= 15) & out
    v = r[sel].mean()
    return float(0.0 if (v is None or not np.isfinite(v)) else v)
