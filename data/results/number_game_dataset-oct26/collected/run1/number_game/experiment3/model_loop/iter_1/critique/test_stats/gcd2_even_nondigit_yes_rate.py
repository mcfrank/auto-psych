# name: gcd2_even_nondigit_yes_rate
# description: Among multi-example all-even sets whose gcd is exactly 2, yes-rate for even targets whose final digit matches no example; observed > null means the model (no "even numbers" rule, mutual-similarity penalty) under-generalises to the even-number concept.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    t = df["target"].astype(int)
    info = {}
    for s in ns.unique():
        xs = [int(x) for x in s.split(",")]
        g = 0
        for x in xs:
            g = math.gcd(g, x)
        info[s] = (len(xs) >= 2 and g == 2, set(x % 10 for x in xs))
    ok = ns.map(lambda s: info[s][0]).astype(bool)
    digs = ns.map(lambda s: info[s][1])
    sel = ok & (t % 2 == 0)
    nomatch = pd.Series([ (tt % 10) not in d for tt, d in zip(t[sel], digs[sel]) ], index=t[sel].index, dtype=bool)
    r = df["rating"].astype(float)[sel][nomatch]
    if len(r) == 0:
        return 0.0
    return float(r.mean())
