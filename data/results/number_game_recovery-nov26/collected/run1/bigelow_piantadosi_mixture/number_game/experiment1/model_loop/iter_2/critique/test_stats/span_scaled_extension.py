# name: span_scaled_extension
# description: Yes-rate for targets within 10 outside the range of wide-span sets (span >= 30) minus that for narrow-span sets (span <= 10, multi-example); observed > null means people's out-of-range extension scales with span more than the model's fixed-scale Erlang prior allows.
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = s.str.count(",").to_numpy() + 1
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1); sp = hi - lo
    d = np.maximum(lo - t, t - hi)
    near = (d >= 1) & (d <= 10) & (n >= 2)
    mw = near & (sp >= 30); mn = near & (sp <= 10)
    a = r[mw].mean() if mw.any() else 0.0; b = r[mn].mean() if mn.any() else 0.0
    return float(a - b)
