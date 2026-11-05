# name: exemplar_proximity_inside_span
# description: Among 2+-example sets and targets strictly inside the examples' span, yes rate for targets within 5 of the nearest example minus yes rate for targets more than 10 from every example; observed > null means people show local exemplar proximity the flat interval-similarity account does not produce.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    t = df["target"].astype(int).values.astype(float)
    E = ex.values.astype(float)
    d = np.nanmin(np.abs(E - t[:, None]), axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    inside = (n.values >= 2) & (t > lo) & (t < hi)
    r = df["rating"].values.astype(float)
    near = inside & (d <= 5); far = inside & (d > 10)
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
