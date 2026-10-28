# name: below_vs_above_span_asymmetry
# description: Yes-rate for targets 5-40 below the smallest example minus targets 5-40 above the largest example; observed < null means people generalize less downward than upward (log-like magnitude: equal linear gaps below are larger ratios), which the symmetric linear-distance model fails to produce.
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
    return m((mn - t >= 5) & (mn - t <= 40)) - m((t - mx >= 5) & (t - mx <= 40))
