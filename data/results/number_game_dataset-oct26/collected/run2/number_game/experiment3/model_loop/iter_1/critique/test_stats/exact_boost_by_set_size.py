# name: exact_boost_by_set_size
# description: Exact-example boost (yes-rate on target = an example minus yes-rate on non-example targets within 1-3 of the nearest example) for 4-example sets minus the same boost for 1-3-example sets; observed < null means the fixed-list concept is less credible for larger sets than the model constant list weight implies.
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
    near = ~isex & (nearest >= 1) & (nearest <= 3)
    b4 = m(isex & (k == 4)) - m(near & (k == 4))
    bs = m(isex & (k <= 3)) - m(near & (k <= 3))
    return b4 - bs
