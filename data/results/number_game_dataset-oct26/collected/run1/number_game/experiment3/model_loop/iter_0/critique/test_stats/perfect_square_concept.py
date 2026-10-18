# name: perfect_square_concept
# description: In sets with at least two examples that are all perfect squares, yes-rate on perfect-square targets minus non-square targets; observed > null means the model under-produces generalisation to the 'square numbers' concept.
def test_statistic(df):
    S = {k * k for k in range(1, 11)}
    sets = df["number_set"].astype(str)
    codes, cats = pd.factorize(sets)
    ok = np.array([("," in s) and all(int(x) in S for x in s.split(",")) for s in cats])
    inset = ok[codes]
    t = df["target"].astype(int).to_numpy()
    ts = np.isin(t, list(S))
    r = df["rating"].to_numpy().astype(float)
    a = inset & ts
    b = inset & ~ts
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
