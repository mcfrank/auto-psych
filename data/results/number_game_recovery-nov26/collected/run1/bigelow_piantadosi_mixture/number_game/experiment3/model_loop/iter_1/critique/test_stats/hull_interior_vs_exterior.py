# name: hull_interior_vs_exterior
# description: Among sets with >=2 examples and targets not in the set, yes rate for targets strictly between the smallest and largest example minus yes rate for targets outside that span; observed > null means people fill in the examples' numerical span more (relative to extending beyond it) than the model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    E = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)  # (rows, 4), NaN padded
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    in_set = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    base = (n >= 2) & ~in_set
    inside = base & (t > lo) & (t < hi)
    outside = base & ((t < lo) | (t > hi))
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(y[inside].mean() - y[outside].mean())
