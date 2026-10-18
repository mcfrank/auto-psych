# name: gcd_divisible_minus_not
# description: On 2-4 example sets whose examples share a common divisor >=3, among targets with the same parity as every example, yes rate for targets divisible by that divisor minus those not divisible; observed above null means people infer common-factor/multiples rules that the final-digit model cannot represent.
def test_statistic(df):
    from functools import reduce
    s = df["number_set"].astype(str)
    uniq = s.unique()
    gmap, pmap = {}, {}
    for u in uniq:
        xs = [int(v) for v in u.split(",")]
        gmap[u] = reduce(math.gcd, xs) if len(xs) > 1 else 0
        ps = set(x % 2 for x in xs)
        pmap[u] = ps.pop() if len(ps) == 1 else -1
    g = s.map(gmap).astype(int)
    p = s.map(pmap).astype(int)
    t = df["target"].astype(int)
    base = (g >= 3) & (p == (t % 2))
    div = (t % g.clip(lower=1)) == 0
    a = df.loc[base & div, "rating"]
    b = df.loc[base & ~div, "rating"]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
