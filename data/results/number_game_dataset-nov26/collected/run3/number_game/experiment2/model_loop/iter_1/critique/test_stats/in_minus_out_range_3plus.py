# name: in_minus_out_range_3plus
# description: For sets with 3+ examples, non-example-target yes rate inside [min,max] of the examples minus outside it; observed above null means people confine generalization to the examples' range more than the model (below = the model over-confines, e.g. via its range-size prior).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    d = df[n >= 3]
    sets = d["number_set"].unique()
    lo = {}; hi = {}; exs = {}
    for s in sets:
        xs = [int(v) for v in str(s).split(",")]
        lo[s] = min(xs); hi[s] = max(xs); exs[s] = set(xs)
    t = d["target"].astype(int)
    a = d["number_set"].map(lo); b = d["number_set"].map(hi)
    isex = pd.Series([tt in exs[s] for s, tt in zip(d["number_set"].values, t.values)], index=d.index)
    inside = (t >= a) & (t <= b) & ~isex
    outside = ((t < a) | (t > b))
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(d.loc[inside, "rating"].mean() - d.loc[outside, "rating"].mean())
