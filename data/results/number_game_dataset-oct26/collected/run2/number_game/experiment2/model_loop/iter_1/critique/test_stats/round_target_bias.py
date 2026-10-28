# name: round_target_bias
# description: In sets with no example divisible by 5, yes-rate for multiple-of-10 targets minus yes-rate for all other non-example targets; observed above the null means people have a set-independent preference for round numbers that the model (similarity + set-derived rules only) under-produces, below means people avoid round targets more than predicted.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    ok = {s: all(int(v) % 5 != 0 for v in s.split(",")) for s in sets}
    sel = ns.map(ok).values.astype(bool)
    t = df["target"].astype(int).values
    padded = ("," + ns + ",")
    tstr = df["target"].astype(int).astype(str)
    ident = np.array([("," + a + ",") in b for a, b in zip(tstr.values, padded.values)])
    y = df["rating"].astype(float).values
    base = sel & ~ident
    a = base & (t % 10 == 0)
    b = base & (t % 10 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
