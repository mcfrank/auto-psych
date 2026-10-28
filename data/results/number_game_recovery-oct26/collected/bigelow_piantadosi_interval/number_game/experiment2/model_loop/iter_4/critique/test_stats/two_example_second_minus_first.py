# name: two_example_second_minus_first
# description: In 2-example sets with examples at least 10 apart, among non-example targets within 8 of exactly one example, yes rate near the second-presented example minus near the first; observed below the null means the model over-weights the more recent example (or under-weights the first) relative to people, above means people favour the recent example more.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    a1, a2 = {}, {}
    for s in uniq:
        e = [int(x) for x in s.split(",")]
        if len(e) == 2 and abs(e[0] - e[1]) >= 10:
            a1[s], a2[s] = e[0], e[1]
        else:
            a1[s], a2[s] = -1000, -1000
    e1 = sets.map(a1).to_numpy()
    e2 = sets.map(a2).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    d1, d2 = np.abs(t - e1), np.abs(t - e2)
    ok = (e1 > 0) & (d1 > 0) & (d2 > 0)
    near1 = ok & (d1 <= 8) & (d2 > 8)
    near2 = ok & (d2 <= 8) & (d1 > 8)
    if near1.sum() == 0 or near2.sum() == 0:
        return 0.0
    return float(r[near2].mean() - r[near1].mean())
