# name: interval_inside_minus_outside
# description: In sets with 2+ examples, among targets not covered by a shared-factor rule (set gcd < 3 or target not a multiple of it) and not identical to an example, yes-rate for targets strictly inside the examples' [min,max] range minus targets outside it; observed above the null means people use an interval/range hypothesis more than the model's per-example similarity yields.
def test_statistic(df):
    sets = df["number_set"].astype(str).unique()
    info = {}
    for s in sets:
        ex = [int(v) for v in s.split(",")]
        g = 0
        for v in ex:
            g = math.gcd(g, v)
        info[s] = (len(ex), min(ex), max(ex), g, set(ex))
    ns = df["number_set"].astype(str)
    k = ns.map({s: v[0] for s, v in info.items()}).values
    lo = ns.map({s: v[1] for s, v in info.items()}).values
    hi = ns.map({s: v[2] for s, v in info.items()}).values
    g = ns.map({s: v[3] for s, v in info.items()}).values
    t = df["target"].astype(int).values
    ident = np.array([tt in info[s][4] for tt, s in zip(t, ns.values)])
    gs = np.where(g >= 3, g, 1)
    covered = (g >= 3) & (t % gs == 0)
    base = (k >= 2) & ~ident & ~covered
    inside = base & (t > lo) & (t < hi)
    outside = base & ((t < lo) | (t > hi))
    r = df["rating"].astype(float).values
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
