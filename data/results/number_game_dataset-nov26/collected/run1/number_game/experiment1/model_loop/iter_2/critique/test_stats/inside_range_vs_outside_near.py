# name: inside_range_vs_outside_near
# description: Yes-rate for targets strictly inside the examples' [min,max] range (multi-example sets) minus yes-rate for targets outside the range but within 5 of the nearest example; observed > null means people fill in the range more (and penalize stepping outside it more) than the summed-similarity model predicts.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    return ex, t, y

def test_statistic(df):
    ex, t, y = _parse(df)
    n = (~np.isnan(ex)).sum(1)
    lo = np.nanmin(ex, 1); hi = np.nanmax(ex, 1)
    d = np.nanmin(np.abs(ex - t[:, None]), 1)
    inside = (n >= 2) & (t > lo) & (t < hi) & (d > 0)
    out = (n >= 2) & ((t < lo) | (t > hi)) & (d <= 5)
    a = y[inside].mean() if inside.any() else 0.0
    b = y[out].mean() if out.any() else 0.0
    return float(a - b)
