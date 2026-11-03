# name: set_size_inrange_contrast
# description: Among non-example targets strictly inside [min,max] of the examples, yes rate for 2-example sets minus yes rate for 4-example sets; observed > null means adding examples tightens/changes generalisation more than the model's size principle predicts (observed < null: less).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    lo, hi = np.nanmin(E, axis=1), np.nanmax(E, axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    base = (t > lo) & (t < hi) & ~in_set
    a, b = base & (n == 2), base & (n == 4)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
