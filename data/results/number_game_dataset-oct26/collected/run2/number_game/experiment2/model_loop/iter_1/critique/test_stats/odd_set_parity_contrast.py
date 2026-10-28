# name: odd_set_parity_contrast
# description: In sets of 3+ examples that are all odd but share no common factor >=3 and no common final digit, yes-rate for non-example odd targets minus even targets; observed above the null means people abstract an explicit "odd numbers" rule sharper than the model's parity-similarity dimension predicts, below means the model over-weights parity.
def test_statistic(df):
    import math
    from functools import reduce
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    ok = {}
    for s in sets:
        ex = [int(v) for v in s.split(",")]
        ok[s] = (len(ex) >= 3 and all(v % 2 == 1 for v in ex)
                 and reduce(math.gcd, ex) < 3 and len(set(v % 10 for v in ex)) > 1)
    sel = ns.map(ok).values.astype(bool)
    t = df["target"].astype(int).values
    padded = ("," + ns + ",")
    tstr = df["target"].astype(int).astype(str)
    ident = np.array([("," + a + ",") in b for a, b in zip(tstr.values, padded.values)])
    y = df["rating"].astype(float).values
    base = sel & ~ident
    a = base & (t % 2 == 1)
    b = base & (t % 2 == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
