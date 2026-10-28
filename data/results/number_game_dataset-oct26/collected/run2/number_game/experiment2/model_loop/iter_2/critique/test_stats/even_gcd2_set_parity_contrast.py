# name: even_gcd2_set_parity_contrast
# description: In 2+-example sets that are all even with greatest common divisor exactly 2 (so the model's multiples-of-g rule, which needs g>=3, is silent), yes-rate for non-example even targets minus odd targets; observed above the null means people apply an explicit "even numbers" rule sharper than the model's parity-similarity dimension, below means the model over-weights parity.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    flag = {}
    for s in sets:
        ex = [int(v) for v in s.split(",")]
        flag[s] = len(ex) >= 2 and int(np.gcd.reduce(ex)) == 2
    exs = {s: set(int(v) for v in s.split(",")) for s in sets}
    fl = ns.map(flag).values.astype(bool)
    t = df["target"].astype(int).values
    ident = np.fromiter((tt in exs[s] for tt, s in zip(t, ns.values)), bool, len(t))
    base = fl & ~ident
    a, b = base & (t % 2 == 0), base & (t % 2 == 1)
    y = df["rating"].astype(float).values
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
