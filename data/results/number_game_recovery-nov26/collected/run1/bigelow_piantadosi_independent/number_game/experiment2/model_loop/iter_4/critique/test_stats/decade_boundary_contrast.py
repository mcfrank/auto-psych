# name: decade_boundary_contrast
# description: For non-example targets 1-6 away from their nearest example, the distance-stratified (count-weighted) yes rate when the target shares the nearest example's tens digit minus when it is across a decade boundary; observed > null means people treat decades ("the 20s") as categories that the model's smooth intervals/neighbourhoods under-produce, observed < null means the model over-produces a decade effect.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    d = dist.min(axis=1)
    nearest = np.take_along_axis(np.where(valid, ex, np.nan), np.argmin(dist, axis=1)[:, None], axis=1)[:, 0]
    same = np.floor(nearest / 10) == np.floor(t / 10)
    tot, wsum = 0.0, 0.0
    for k in range(1, 7):
        m = d == k
        a, b = m & same, m & ~same
        if a.sum() > 0 and b.sum() > 0:
            w = float(m.sum())
            tot += w * (y[a].mean() - y[b].mean())
            wsum += w
    return float(tot / wsum) if wsum > 0 else 0.0
