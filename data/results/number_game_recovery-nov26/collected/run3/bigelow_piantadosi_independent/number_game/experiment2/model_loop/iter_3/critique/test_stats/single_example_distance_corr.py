# name: single_example_distance_corr
# description: Among single-example sets, Pearson correlation between yes and |target - example| for non-example targets; observed more negative than null means people's yes rate falls off with numeric distance more steeply than the model, less negative means the model's distance gradient is too steep.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = (~np.isnan(ex)).sum(1)
    is_ex = (ex == t[:, None]).any(1)
    P = set(n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1)))
    def m(mask):
        return float(y[mask].mean()) if mask.sum() > 0 else 0.0
    sel = (k == 1) & ~is_ex
    if sel.sum() < 3:
        return 0.0
    dd = np.abs(t[sel] - ex[sel, 0]); yy = y[sel]
    if dd.std() == 0 or yy.std() == 0:
        return 0.0
    return float(np.corrcoef(dd, yy)[0, 1])
