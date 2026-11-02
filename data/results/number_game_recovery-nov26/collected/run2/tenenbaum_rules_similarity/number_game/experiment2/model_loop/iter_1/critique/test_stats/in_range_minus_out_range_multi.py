# name: in_range_minus_out_range_multi
# description: Among sets with 2+ examples and targets not in the set, yes rate for targets inside [min,max] of the examples minus yes rate for targets outside that span; observed < null means the model overweights the convex range (interval) hypothesis relative to people, observed > null means people respect the span more sharply than the model.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    n = (~np.isnan(ex)).sum(axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    inset = (ex == t[:, None]).any(axis=1)
    r = df["rating"].to_numpy().astype(float)
    base = (n >= 2) & ~inset
    inside = base & (t >= lo) & (t <= hi)
    outside = base & ((t < lo) | (t > hi))
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
