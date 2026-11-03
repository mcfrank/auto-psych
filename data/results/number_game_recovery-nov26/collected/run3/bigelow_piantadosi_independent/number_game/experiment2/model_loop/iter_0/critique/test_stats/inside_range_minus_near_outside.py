# name: inside_range_minus_near_outside
# description: For multi-example sets, yes rate on non-example targets strictly between the smallest and largest example, minus yes rate on targets within 10 outside that span; observed above null means people fill in the examples' span more sharply than the model's intervals predict (the model extrapolates too much), and below null means the model fills in too much.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    mind = ex.sub(t, axis=0).abs().min(axis=1)
    n = ex.notna().sum(axis=1)
    lo = ex.min(axis=1)
    hi = ex.max(axis=1)
    inside = ((t > lo) & (t < hi) & (mind > 0) & (n > 1)).values
    outside = (((t < lo) | (t > hi)) & (mind <= 10) & (n > 1)).values
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
