# name: inrange_nearest_example_gap
# description: Multi-example sets, non-example targets strictly inside [min,max]: yes-rate for targets within 5 of the nearest example minus yes-rate for targets more than 15 from every example; the model gives all in-range targets full magnitude credit, so observed above null means people use proximity to individual examples (exemplar similarity) that the model misses, below means they avoid near neighbours.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    isex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    d = np.nanmin(np.abs(E - t[:, None]), axis=1)
    ok = (n >= 2) & ~isex & (t > lo) & (t < hi)
    a = ok & (d <= 5); b = ok & (d > 15)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
