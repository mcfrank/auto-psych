# name: power_of_two_set_contrast
# description: In multi-example sets whose examples are all powers of two, yes-rate for power-of-two targets minus other targets (identity trials excluded); observed above the null means people abstract a "powers of 2" rule the model (which only has a multiples-of-g rule and graded factor similarity) under-produces.
def test_statistic(df):
    p2 = {1, 2, 4, 8, 16, 32, 64}
    sets = df["number_set"].astype(str).unique()
    ok = {}
    for s in sets:
        ex = [int(v) for v in s.split(",")]
        ok[s] = len(ex) >= 2 and all(v in p2 for v in ex)
    ns = df["number_set"].astype(str)
    inset = ns.map(ok).astype(bool).values
    t = df["target"].astype(int).values
    exs = ns.map({s: set(int(v) for v in s.split(",")) for s in sets})
    ident = np.array([tt in e for tt, e in zip(t, exs)])
    tp2 = np.isin(t, list(p2))
    r = df["rating"].astype(float).values
    a = inset & ~ident & tp2
    b = inset & ~ident & ~tp2
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
