# name: above_vs_below_range_asymmetry
# description: Among all sets, for targets sharing no final digit with any example and not multiples of a gcd>=3, yes rate for targets above the largest example minus targets below the smallest example; observed > null means people generalise upward more than the model's symmetric decade-band familiarity predicts (observed < null: downward).
def test_statistic(df):
    import math
    from functools import reduce
    sets = df["number_set"].unique()
    rows = {}
    for s in sets:
        xs = [int(x) for x in s.split(",")]
        dig = 0
        for x in xs:
            dig |= 1 << (x % 10)
        rows[s] = (reduce(math.gcd, xs), min(xs), max(xs), dig)
    ns = df["number_set"]
    gg = ns.map({k: v[0] for k, v in rows.items()}).values
    lo = ns.map({k: v[1] for k, v in rows.items()}).values
    hi = ns.map({k: v[2] for k, v in rows.items()}).values
    dig = ns.map({k: v[3] for k, v in rows.items()}).values.astype(np.int64)
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    mult = (gg >= 3) & (t % np.maximum(gg, 1) == 0)
    share_dig = ((dig >> (t % 10)) & 1) == 1
    base = ~mult & ~share_dig
    a = base & (t > hi)
    b = base & (t < lo)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
