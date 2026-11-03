# name: set_size_generalization_slope
# description: OLS slope of the non-example yes rate (per number-of-examples level, 1-4) on the number of examples; t_observed above null_mean (less negative) means generalization tightens less with more examples than the tempered size principle predicts, below means it tightens more.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    v = ~np.isnan(s); n = v.sum(axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].to_numpy(dtype=float)
    sv = np.where(v, s, np.nan)
    is_ex = (np.where(v, s, -999) == t[:, None]).any(axis=1)
    d = np.nanmin(np.abs(sv - t[:, None]), axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    base = ~is_ex
    xs = []; ys = []
    for k in range(1, 5):
        m = base & (n == k)
        if m.sum() > 0:
            xs.append(k); ys.append(r[m].mean())
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.array(xs, float), np.array(ys), 1)[0])
