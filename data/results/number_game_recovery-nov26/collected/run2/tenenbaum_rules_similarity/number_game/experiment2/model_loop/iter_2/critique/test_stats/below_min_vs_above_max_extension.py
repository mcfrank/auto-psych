# name: below_min_vs_above_max_extension
# description: Among sets of 2+ examples, yes rate for targets 1-15 below the smallest example minus yes rate for targets 1-15 above the largest example; observed above the null means people extend downward more readily than the model's symmetric intervals/rules predict (below = they extend upward more).
def test_statistic(df):
    exm = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    lo = np.nanmin(exm, axis=1)
    hi = np.nanmax(exm, axis=1)
    n = (~np.isnan(exm)).sum(axis=1)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy()
    a = (n >= 2) & (t < lo) & (lo - t <= 15)
    b = (n >= 2) & (t > hi) & (t - hi <= 15)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
