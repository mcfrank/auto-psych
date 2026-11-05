# name: far_opposite_parity_unsupported_yes
# description: For sets whose examples all share parity, yes rate of opposite-parity non-example targets lying more than 5 outside the example range and in no small rule (<=25 members, of the model's 45) containing all examples; observed above null means the parity-restricted model now rejects far opposite-parity targets too firmly (people keep a floor), below null means people reject them even more firmly.
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
        if len({e % 2 for e in ex}) != 1:
            continue
        lo, hi = min(ex), max(ex)
        cons = set()
        for c in R.values():
            if len(c) <= 25 and all(e in c for e in ex):
                cons |= c
        t = g["target"].to_numpy()
        m = (t % 2 != ex[0] % 2) & ((t < lo - 5) | (t > hi + 5)) & ~np.isin(t, list(cons))
        vals.append(g["rating"].to_numpy()[m])
    v = np.concatenate(vals) if vals else np.array([])
    return float(v.mean()) if len(v) else 0.0
