# name: wide_vs_narrow_span_fill
# description: Multi-example sets: yes-rate for targets inside [min,max] when span>=40 minus when span<=15; observed below null means people fill wide spans less than the model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    inside = (t >= lo) & (t <= hi) & (n >= 2)
    span = hi - lo
    return rate(inside & (span >= 40)) - rate(inside & (span <= 15))
