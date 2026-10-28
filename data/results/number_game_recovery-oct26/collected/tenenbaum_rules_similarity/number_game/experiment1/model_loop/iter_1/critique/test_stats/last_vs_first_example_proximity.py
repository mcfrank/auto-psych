# name: last_vs_first_example_proximity
# description: Among multi-example sets, targets not in set and within 15 of exactly one of the first/last example: yes rate (near last shown) minus yes rate (near first shown); model is order-invariant, so observed > null_mean means recency-weighted generalization.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(s, axis=1); mx = np.nanmax(s, axis=1)
    n = np.sum(~np.isnan(s), axis=1)
    idx_last = n - 1
    first = s[:, 0]
    last = s[np.arange(len(s)), idx_last]
    dfst = np.abs(t - first); dlst = np.abs(t - last)
    notin = ~np.any(s == t[:, None], axis=1)
    base = (n >= 2) & notin & (np.abs(first - last) > 30)
    nl = base & (dlst <= 15) & (dfst > 15)
    nf = base & (dfst <= 15) & (dlst > 15)
    if not nl.any() or not nf.any():
        return 0.0
    return float(r[nl].mean() - r[nf].mean())
