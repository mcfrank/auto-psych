# name: nearest_first_minus_nearest_last
# description: For sets of 3-4 examples, yes-rate for non-example targets whose nearest example is the first one shown minus that for targets whose nearest example is the last one shown; observed above the null means people show stronger primacy than the anchoring model produces, below means weaker primacy or a recency advantage the model misses.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    k = (~np.isnan(ex)).sum(axis=1)
    t = df["target"].astype(float).to_numpy()
    d = np.abs(ex - t[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(d, axis=1)
    dmin = d.min(axis=1)
    # unique nearest only (no ties)
    ties = (d == dmin[:, None]).sum(axis=1) > 1
    base = (k >= 3) & (dmin > 0) & ~ties
    first = base & (nearest == 0)
    last = base & (nearest == k - 1)
    r = df["rating"].astype(float).to_numpy()
    if first.sum() == 0 or last.sum() == 0:
        return 0.0
    return float(r[first].mean() - r[last].mean())
