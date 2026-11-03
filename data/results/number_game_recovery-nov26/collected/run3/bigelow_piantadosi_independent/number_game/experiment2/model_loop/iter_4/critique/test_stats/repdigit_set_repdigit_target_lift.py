# name: repdigit_set_repdigit_target_lift
# description: In sets containing >=2 repdigit examples (11,22,...,99), yes rate on non-example repdigit targets minus yes rate on other non-example targets; observed > null means the model lacks a "repeated digits" concept people use.
def test_statistic(df):
    rep = {11, 22, 33, 44, 55, 66, 77, 88, 99}
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    feat = {}
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        feat[s] = (sum(v in rep for v in ex) >= 2, ex)
    ok = sets.map(lambda s: feat[s][0]).to_numpy(bool)
    isex = np.array([ti in feat[s][1] for s, ti in zip(sets, t)])
    base = ok & ~isex
    isrep = np.isin(t, list(rep))
    a = base & isrep; b = base & ~isrep
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
