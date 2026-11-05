# name: broad_rule_inside_range_member_yes
# description: For sets of >=3 examples whose smallest consistent rule (of the model's 45) is broad (>=25 members, e.g. even, odd, multiples of 3), yes rate of non-example members of that rule lying inside the example range; observed below null means people endorse well-supported members of broad concepts less than the model predicts (a ceiling well below 1), above null means more.
def _rules():
    N = range(1, 101)
    r = {"odd": {n for n in N if n % 2}, "even": {n for n in N if n % 2 == 0},
         "primes": {n for n in N if n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))},
         "squares": {n * n for n in range(1, 11)}, "cubes": {n ** 3 for n in range(1, 5)},
         "pow2no32": {2, 4, 8, 16, 64}, "pow2or37": {2, 4, 8, 16, 32, 37, 64}}
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        r["mult%d" % m] = set(range(m, 101, m))
    for m in range(2, 11):
        r["pow%d" % m] = {m ** k for k in range(1, 8) if m ** k <= 100}
    for d in range(10):
        r["ends%d" % d] = {n for n in N if n % 10 == d}
        r["contains%d" % d] = {n for n in N if str(d) in str(n)}
    return r

def test_statistic(df):
    R = _rules()
    d = df[["number_set", "target", "rating"]].copy()
    d["target"] = d["target"].astype(int)
    vals = []
    for s, g in d.groupby("number_set"):
        ex = [int(x) for x in s.split(",")]
        if len(ex) < 3:
            continue
        cons = [c for c in R.values() if all(e in c for e in ex)]
        if not cons:
            continue
        best = min(cons, key=len)
        if len(best) < 25:
            continue
        lo, hi = min(ex), max(ex)
        t = g["target"].to_numpy()
        m = np.isin(t, list(best)) & ~np.isin(t, ex) & (t >= lo) & (t <= hi)
        vals.append(g["rating"].to_numpy()[m])
    v = np.concatenate(vals) if vals else np.array([])
    return float(v.mean()) if len(v) else 0.0
