# name: outside_range_minus_inside_gap
# description: For multi-example sets, yes rate for non-example targets within 10 outside the example range [min,max] minus yes rate for non-example targets strictly inside the range; observed > null (less negative) means people extend generalization beyond the example range more than the model's interval hypotheses do (softer range edges).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    k = (~np.isnan(ex)).sum(axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    notex = ~(ex == t[:, None]).any(axis=1)
    r = df["rating"].to_numpy().astype(float)
    inside = (k >= 2) & notex & (t > lo) & (t < hi)
    outside = (k >= 2) & notex & (((t < lo) & (t >= lo - 10)) | ((t > hi) & (t <= hi + 10)))
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[outside].mean() - r[inside].mean())
