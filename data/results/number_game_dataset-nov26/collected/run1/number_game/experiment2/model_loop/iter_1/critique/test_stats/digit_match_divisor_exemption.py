# name: digit_match_divisor_exemption
# description: In multi-example sets whose examples' gcd is >=3, among targets sharing some example's final digit (target not an example), yes-rate for gcd-multiples minus yes-rate for non-multiples; observed < null means people do NOT apply the common-divisor penalty to final-digit-matching targets (the digit rule overrides it, e.g. 19/59 for "69,9,39,21") while the model's constant divisor penalty does, observed > null means the divisor penalty is even stronger for them.
def test_statistic(df):
    from math import gcd
    from functools import reduce
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    info = {s: [int(v) for v in s.split(",")] for s in uniq}
    g = sets.map({s: (reduce(gcd, l) if len(l) > 1 else 0) for s, l in info.items()}).to_numpy()
    digs = sets.map({s: frozenset(e % 10 for e in l) for s, l in info.items()})
    exs = sets.map({s: frozenset(l) for s, l in info.items()})
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    dm = np.array([(ti % 10) in dg for ti, dg in zip(t, digs)])
    isex = np.array([ti in e for ti, e in zip(t, exs)])
    base = (g >= 3) & dm & ~isex
    div = base & (t % np.maximum(g, 1) == 0)
    nd = base & (t % np.maximum(g, 1) != 0)
    if div.sum() == 0 or nd.sum() == 0:
        return 0.0
    return float(r[div].mean() - r[nd].mean())
