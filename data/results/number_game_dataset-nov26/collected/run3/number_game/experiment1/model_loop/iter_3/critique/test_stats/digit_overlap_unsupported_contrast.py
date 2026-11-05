# name: digit_overlap_unsupported_contrast
# description: Among non-example targets in no small rule (<=25 members) containing all examples, yes rate of targets sharing at least one digit character (any position) with some example minus targets sharing none; observed above null means people use surface digit similarity (e.g. 61, 19, 12 for 100,1,16) beyond the model's rules and intervals, below null means less.
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
    a, b = [], []
    for s, g in d.groupby("number_set"):
        ex = [int(x) for x in s.split(",")]
        small = set()
        for c in R.values():
            if len(c) <= 25 and all(e in c for e in ex):
                small |= c
        digs = set("".join(str(e) for e in ex))
        t = g["target"].to_numpy()
        r = g["rating"].to_numpy()
        ok = ~np.isin(t, ex) & ~np.isin(t, list(small))
        share = np.array([bool(set(str(x)) & digs) for x in t], dtype=bool)
        a.append(r[ok & share]); b.append(r[ok & ~share])
    a, b = np.concatenate(a), np.concatenate(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
