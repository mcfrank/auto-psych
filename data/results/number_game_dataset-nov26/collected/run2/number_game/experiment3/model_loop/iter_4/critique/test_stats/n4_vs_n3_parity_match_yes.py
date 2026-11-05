# name: n4_vs_n3_parity_match_yes
# description: Among sets whose examples all share one parity, yes rate for parity-matching non-member targets on four-example sets minus that on three-example sets; observed < null means people narrow generalization from 3 to 4 examples more than the memory-drop model allows (observed > null: the model narrows too much at four examples).
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    ex = [[int(x) for x in s.split(",")] for s in u]
    nn = np.array([len(e) for e in ex])
    par = np.array([e[0] % 2 if len({x % 2 for x in e}) == 1 else -1 for e in ex])
    mem = np.zeros((len(u), 101), bool)
    for i, e in enumerate(ex):
        mem[i, e] = True
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    ok = (par[inv] >= 0) & (t % 2 == par[inv]) & ~mem[inv, t]
    a = ok & (nn[inv] == 4)
    b = ok & (nn[inv] == 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
