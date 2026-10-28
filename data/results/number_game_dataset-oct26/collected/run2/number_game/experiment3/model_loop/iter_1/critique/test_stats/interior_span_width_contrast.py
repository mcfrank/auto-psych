# name: interior_span_width_contrast
# description: Yes-rate for non-example targets strictly inside the span of 2+ examples when the span is >=50 minus when it is <=25; observed > null means people fill in wide spans more readily than the model (which treats all interior targets as zero size-mismatch), observed < null means less readily.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    mn = ex.min(axis=1).to_numpy(); mx = ex.max(axis=1).to_numpy()
    k = ex.notna().sum(axis=1).to_numpy()
    first = ex[0].to_numpy(); last = ex.ffill(axis=1).iloc[:, -1].to_numpy()
    isex = ex.eq(pd.Series(t, index=ex.index), axis=0).any(axis=1).to_numpy()
    nearest = np.nanmin(np.abs(ex.to_numpy() - t[:, None]), axis=1)
    gap = np.maximum(np.maximum(mn - t, t - mx), 0)
    def m(mask):
        return float(y[mask].mean()) if mask.sum() > 0 else 0.0
    sel = (k >= 2) & (gap == 0) & ~isex
    span = mx - mn
    return m(sel & (span >= 50)) - m(sel & (span <= 25))
