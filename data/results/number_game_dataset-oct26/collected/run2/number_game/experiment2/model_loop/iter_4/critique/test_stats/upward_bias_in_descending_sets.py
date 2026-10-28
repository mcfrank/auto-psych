# name: upward_bias_in_descending_sets
# description: In 2+-example sets presented in descending order (last < first), yes-rate for targets 1-20 above the largest example minus targets 1-20 below the smallest; observed above the null means people keep a fixed upward (larger-number) generalization bias even when the sequence moves downward, which the model's purely order-directed momentum (predicting a downward advantage here) under-produces (below means momentum is stronger than modelled).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    ex = {s: [int(v) for v in s.split(",")] for s in sets}
    desc = ns.map({s: len(e) >= 2 and e[-1] < e[0] for s, e in ex.items()}).values.astype(bool)
    hi = ns.map({s: max(e) for s, e in ex.items()}).values
    lo = ns.map({s: min(e) for s, e in ex.items()}).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    up = desc & (t > hi) & (t - hi <= 20)
    dn = desc & (t < lo) & (lo - t <= 20)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(y[up].mean() - y[dn].mean())
