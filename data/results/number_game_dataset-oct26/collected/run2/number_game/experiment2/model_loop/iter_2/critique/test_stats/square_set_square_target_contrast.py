# name: square_set_square_target_contrast
# description: In 2+-example sets whose examples are all perfect squares, yes-rate for non-example perfect-square targets minus non-square targets; observed above the null means people abstract a "square numbers" rule the model (no squareness feature) under-produces, below means the model over-produces it via factor similarity.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    sq = set(i * i for i in range(1, 11))
    flag = {s: (len(s.split(",")) >= 2 and all(int(v) in sq for v in s.split(","))) for s in sets}
    exs = {s: set(int(v) for v in s.split(",")) for s in sets}
    fl = ns.map(flag).values.astype(bool)
    t = df["target"].astype(int).values
    ident = np.fromiter((tt in exs[s] for tt, s in zip(t, ns.values)), bool, len(t))
    tsq = np.isin(t, list(sq))
    base = fl & ~ident
    a, b = base & tsq, base & ~tsq
    y = df["rating"].astype(float).values
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
