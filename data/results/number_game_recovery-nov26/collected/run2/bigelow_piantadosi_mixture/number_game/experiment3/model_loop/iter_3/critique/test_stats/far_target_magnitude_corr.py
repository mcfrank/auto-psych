# name: far_target_magnitude_corr
# description: Correlation between target value and rating among non-example targets more than 15 from every example; observed above null means people say yes to distant large numbers more (relative to distant small ones) than the model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    nd = x.sub(t, axis=0).abs().min(axis=1)
    sel = (nd > 15) & ~isin
    a = t[sel].astype(float)
    b = r[sel]
    v = float(np.corrcoef(a, b)[0, 1]) if sel.sum() > 2 and b.std() > 0 else 0.0
    return float(0.0 if (v is None or not np.isfinite(v)) else v)
