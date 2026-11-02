# name: interior_proximity_gradient
# description: For sets of >=2 examples, among non-example targets strictly inside [min,max], yes rate for targets within 3 of the nearest example minus targets more than 10 from every example; observed > null means people generalise by similarity to individual examples inside the span (a gradient the model's whole-interval hypotheses, near-flat inside the span, under-produce), observed < null the reverse.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    lo, hi = np.nanmin(E, axis=1), np.nanmax(E, axis=1)
    near = np.nanmin(np.abs(E - t[:, None]), axis=1)
    base = (n >= 2) & (t > lo) & (t < hi) & (near >= 1)
    a, b = base & (near <= 3), base & (near > 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
