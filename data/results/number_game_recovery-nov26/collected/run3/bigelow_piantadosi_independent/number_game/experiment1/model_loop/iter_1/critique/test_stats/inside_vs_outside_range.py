# name: inside_vs_outside_range
# description: Yes rate for targets strictly inside the [min,max] span of the examples minus yes rate for targets outside that span (sets with 2+ examples); observed > null means people treat the example range as a region more than the model does.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    r = df["rating"].astype(float).to_numpy()
    inside = (n >= 2) & (t > lo) & (t < hi)
    outside = (n >= 2) & ((t < lo) | (t > hi))
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
