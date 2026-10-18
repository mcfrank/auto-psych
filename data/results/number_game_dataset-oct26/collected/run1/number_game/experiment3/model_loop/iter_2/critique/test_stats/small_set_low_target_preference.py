# name: small_set_low_target_preference
# description: Among sets whose examples are all <=10, for targets that are not multiples of a gcd>=3, yes rate on targets 11-30 minus yes rate on targets 51-100; observed > null means people favour nearby-but-outside small targets over far large ones more than the model's decade-band familiarity predicts, observed < null the reverse.
def test_statistic(df):
    import math
    from functools import reduce
    sets = df["number_set"].unique()
    rows = {s: (reduce(math.gcd, [int(x) for x in s.split(",")]), max(int(x) for x in s.split(","))) for s in sets}
    ns = df["number_set"]
    gg = ns.map({k: v[0] for k, v in rows.items()}).values
    mx = ns.map({k: v[1] for k, v in rows.items()}).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    mult = (gg >= 3) & (t % np.maximum(gg, 1) == 0)
    base = (mx <= 10) & ~mult
    a = base & (t >= 11) & (t <= 30)
    b = base & (t >= 51)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
