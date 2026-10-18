# name: factor_multiple_inside_vs_outside_range
# description: Among multi-example sets with gcd>=3, for targets that are multiples of the gcd, yes rate when the target lies within [min,max] of the examples minus when it lies outside; observed > null means people restrict the factor rule to the examples' range more than the model (which pre-empts magnitude under a rule) predicts, observed < null the reverse.
def test_statistic(df):
    import math
    from functools import reduce
    sets = df["number_set"].unique()
    info = {}
    for s in sets:
        xs = [int(x) for x in s.split(",")]
        info[s] = (reduce(math.gcd, xs), min(xs), max(xs), len(xs))
    gg = df["number_set"].map({k: v[0] for k, v in info.items()}).values
    lo = df["number_set"].map({k: v[1] for k, v in info.items()}).values
    hi = df["number_set"].map({k: v[2] for k, v in info.items()}).values
    n = df["number_set"].map({k: v[3] for k, v in info.items()}).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    mult = (n >= 2) & (gg >= 3) & (t % np.maximum(gg, 1) == 0)
    inside = (t >= lo) & (t <= hi)
    a = mult & inside
    b = mult & ~inside
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
