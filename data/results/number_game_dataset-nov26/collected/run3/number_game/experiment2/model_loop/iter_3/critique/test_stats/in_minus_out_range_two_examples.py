# name: in_minus_out_range_two_examples
# description: For two-example sets, non-example-target yes rate for targets strictly inside [min,max] of the examples minus targets outside it; observed above null means people confine generalization to the examples' range more than the model does for pairs (below = model over-confines).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    sub = df[n == 2]
    parts = sub["number_set"].str.split(",", expand=True).astype(int)
    lo = parts.min(axis=1); hi = parts.max(axis=1)
    t = sub["target"].astype(int)
    nonex = (t != parts[0]) & (t != parts[1])
    inside = nonex & (t > lo) & (t < hi)
    outside = nonex & ((t < lo) | (t > hi))
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    r = sub["rating"]
    return float(r[inside].mean() - r[outside].mean())
