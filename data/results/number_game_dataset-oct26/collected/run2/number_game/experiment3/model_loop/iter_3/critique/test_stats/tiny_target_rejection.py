# name: tiny_target_rejection
# description: Among sets whose smallest example is >= 15, yes-rate for targets 1-3 minus targets 4-10; observed < null means people reject the very smallest numbers (1, 2, 3) more than graded size similarity predicts (model over-produces yes on them).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(ex)), k - 1]
    mn = np.nanmin(ex, axis=1)
    mx = np.nanmax(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    m = mn >= 15
    return rate(m & (t <= 3)) - rate(m & (t >= 4) & (t <= 10))
