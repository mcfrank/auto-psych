# name: range_boundary_step
# description: Among non-example targets in no small rule (<=25 members) containing all examples, yes rate of targets just inside the example range (within 5 of min or max) minus targets just outside it (within 5 beyond min or max); observed below null means people's generalization falls off more smoothly at the range edge than the model's interval hypotheses (a sharp step) predict, above null means an even sharper edge.
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
    ins, outs = [], []
    for s, g in d.groupby("number_set"):
        ex = [int(x) for x in s.split(",")]
        lo, hi = min(ex), max(ex)
        small = set()
        for c in R.values():
            if len(c) <= 25 and all(e in c for e in ex):
                small |= c
        t = g["target"].to_numpy()
        r = g["rating"].to_numpy()
        ok = ~np.isin(t, ex) & ~np.isin(t, list(small))
        inside = ok & (t >= lo) & (t <= hi) & ((t - lo <= 5) | (hi - t <= 5))
        outside = ok & (((t < lo) & (lo - t <= 5)) | ((t > hi) & (t - hi <= 5)))
        ins.append(r[inside]); outs.append(r[outside])
    a, b = np.concatenate(ins), np.concatenate(outs)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
