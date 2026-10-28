# name: log_magnitude_region_gap
# description: Among out-of-span targets with gap 5-40 from the span, yes-rate when the nearest bounding example is <=20 minus when it is >=60; observed < null means equal linear gaps hurt more among small numbers (log-like compression), which the model under-produces.
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
    sel = (gap >= 5) & (gap <= 40)
    bound = np.where(t < mn, mn, mx)
    return m(sel & (bound <= 20)) - m(sel & (bound >= 60))
