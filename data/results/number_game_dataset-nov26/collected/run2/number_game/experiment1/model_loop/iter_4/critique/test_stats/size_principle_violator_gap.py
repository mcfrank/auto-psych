# name: size_principle_violator_gap
# description: Among parity-homogeneous sets, yes-rate for parity-VIOLATING targets in 3-4-example sets minus in 2-example sets; observed < null means people reject rule violators more sharply as examples accumulate (a size-principle effect the model lacks), observed > null the reverse.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = pd.Series(sets.unique())
    par = {}
    for s in uniq:
        ex = [int(x) for x in s.split(",")]
        r = {e % 2 for e in ex}
        par[s] = (len(ex), r.pop() if len(r) == 1 else -1)
    n = sets.map(lambda s: par[s][0]).to_numpy()
    p = sets.map(lambda s: par[s][1]).to_numpy()
    t = df["target"].astype(int).to_numpy() % 2
    y = df["rating"].astype(float).to_numpy()
    viol = (p >= 0) & (t != p)
    big = viol & (n >= 3)
    small = viol & (n == 2)
    if big.sum() == 0 or small.sum() == 0:
        return 0.0
    return float(y[big].mean() - y[small].mean())
