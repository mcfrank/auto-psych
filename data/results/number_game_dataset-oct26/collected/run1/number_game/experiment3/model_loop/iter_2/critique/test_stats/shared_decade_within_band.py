# name: shared_decade_within_band
# description: Among multi-example sets with no explicit rule (gcd<3 and not all same final digit), for targets inside the examples' decade band that share no final digit with any example, yes rate when the target shares a tens digit with some example minus when it does not; observed > null means people use specific-decade similarity that the model's flat within-band familiarity misses.
def test_statistic(df):
    import math
    from functools import reduce
    sets = df["number_set"].unique()
    rows = {}
    for s in sets:
        xs = [int(x) for x in s.split(",")]
        g = reduce(math.gcd, xs)
        uni = len({x % 10 for x in xs}) == 1
        rule = len(xs) >= 2 and (g >= 3 or uni)
        dec = 0
        dig = 0
        for x in xs:
            dec |= 1 << (x // 10)
            dig |= 1 << (x % 10)
        rows[s] = (len(xs) >= 2 and not rule, min(x // 10 for x in xs), max(x // 10 for x in xs), dec, dig)
    ns = df["number_set"]
    ok = ns.map({k: v[0] for k, v in rows.items()}).values.astype(bool)
    lo = ns.map({k: v[1] for k, v in rows.items()}).values
    hi = ns.map({k: v[2] for k, v in rows.items()}).values
    dec = ns.map({k: v[3] for k, v in rows.items()}).values.astype(np.int64)
    dig = ns.map({k: v[4] for k, v in rows.items()}).values.astype(np.int64)
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    td = t // 10
    share_dec = ((dec >> td) & 1) == 1
    share_dig = ((dig >> (t % 10)) & 1) == 1
    base = ok & (td >= lo) & (td <= hi) & ~share_dig
    a = base & share_dec
    b = base & ~share_dec
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
