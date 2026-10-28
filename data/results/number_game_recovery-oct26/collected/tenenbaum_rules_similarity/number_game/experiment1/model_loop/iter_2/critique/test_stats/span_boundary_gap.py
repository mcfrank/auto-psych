# name: span_boundary_gap
# description: For sets of 2+ examples spanning at least 10, yes rate for non-example targets strictly inside the range minus targets 1-10 outside it; the interval mechanism makes the range edge a sharp boundary, so observed below null_mean means people's generalization crosses the span edge more smoothly (similarity-like) than the model, above means a sharper edge.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",")
    ex = pd.DataFrame(sets.tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    n_ex = np.sum(~np.isnan(ex), axis=1)
    outside = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    base = (n_ex >= 2) & ((hi - lo) >= 10) & (~member)
    a = base & (outside == 0); b = base & (outside >= 1) & (outside <= 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
