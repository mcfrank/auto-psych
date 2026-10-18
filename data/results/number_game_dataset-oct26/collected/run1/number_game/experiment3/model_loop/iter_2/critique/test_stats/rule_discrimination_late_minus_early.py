# name: rule_discrimination_late_minus_early
# description: For multi-example rule sets (gcd>=3 or all examples share a final digit), the rule-consistent minus rule-inconsistent yes-rate gap in late trials (trial_index above median) minus that gap in early trials; observed > null means people become more rule-bound over the session (observed < null: less), a change the stationary model cannot produce.
def test_statistic(df):
    import math
    from functools import reduce
    sets = df["number_set"].unique()
    rows = {}
    for s in sets:
        xs = [int(x) for x in s.split(",")]
        g = reduce(math.gcd, xs)
        uni = len({x % 10 for x in xs}) == 1
        rows[s] = (len(xs) >= 2 and (g >= 3 or uni), g, xs[0] % 10 if uni else -1)
    ns = df["number_set"]
    isrule = ns.map({k: v[0] for k, v in rows.items()}).values.astype(bool)
    gg = ns.map({k: v[1] for k, v in rows.items()}).values
    ud = ns.map({k: v[2] for k, v in rows.items()}).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    ti = df["trial_index"].values
    cons = ((gg >= 3) & (t % np.maximum(gg, 1) == 0)) | (t % 10 == ud)
    late = ti > np.median(ti)
    out = []
    for half in (late, ~late):
        a = isrule & half & cons
        b = isrule & half & ~cons
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        out.append(r[a].mean() - r[b].mean())
    return float(out[0] - out[1])
