# name: powers_of_two_concept
# description: In sets whose examples are all powers of two (2..64), yes-rate on power-of-two targets minus yes-rate on other targets; observed > null means the model under-produces generalisation to the 'powers of two' mathematical concept.
def test_statistic(df):
    P = {1, 2, 4, 8, 16, 32, 64}
    sets = df["number_set"].astype(str)
    codes, cats = pd.factorize(sets)
    ok = np.array([all(int(x) in P for x in s.split(",")) and ("," in s) for s in cats])
    inset = ok[codes]
    t = df["target"].astype(int).to_numpy()
    tp = np.isin(t, list(P))
    r = df["rating"].to_numpy().astype(float)
    a = inset & tp
    b = inset & ~tp
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
