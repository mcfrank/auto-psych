# name: power2_set_even_generalization
# description: Yes rate on even, non-power-of-two, non-example targets for sets of two or more examples that are all powers of two; observed above null means people generalize from power-of-two examples to "even numbers" more than the (tempered) size principle allows, below means they lock onto powers of two more tightly.
def test_statistic(df):
    p2 = {1, 2, 4, 8, 16, 32, 64}
    sets = df["number_set"].astype(str)
    ok = {}
    for s in sets.unique():
        a = [int(x) for x in s.split(",")]
        ok[s] = len(a) >= 2 and all(e in p2 for e in a)
    m = sets.map(ok).to_numpy().astype(bool)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    notp2 = ~np.isin(t, list(p2))
    sel = m & (t % 2 == 0) & notp2
    return float(r[sel].mean()) if sel.any() else 0.0
