# name: contains_digit_only_rule_member_yes
# description: For sets where some "contains digit d" rule includes all examples but no "ends in" rule does (examples do not share a last digit), yes rate of non-example targets containing d; observed below null means the model over-relies on contains-digit concepts (e.g. predicting 8,18,28 for 98,83), above null means people use them more.
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
        if len({e % 10 for e in ex}) == 1:
            continue
        digs = [dd for dd in range(10) if all(e in R["contains%d" % dd] for e in ex)]
        if not digs:
            continue
        t = g["target"].astype(str)
        m = np.zeros(len(g), dtype=bool)
        for dd in digs:
            m |= t.str.contains(str(dd)).to_numpy()
        m &= ~g["target"].isin(ex).to_numpy()
        vals.append(g["rating"].to_numpy()[m])
    v = np.concatenate(vals) if vals else np.array([])
    return float(v.mean()) if len(v) else 0.0
