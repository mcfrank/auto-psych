# name: congruence_beyond_parity
# description: For 3-4-example sets whose sorted differences share a gcd g in {4,6,8,12} with examples not multiples of g (progressions like 41,47,53), yes-rate on non-example targets congruent to the examples mod g minus same-parity non-congruent targets; observed above null_mean means people extract step/progression rules (beyond parity) that the model's fixed rule set under-produces.
def test_statistic(df):
    from functools import reduce
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        a = sorted(int(x) for x in s.split(","))
        g = 0
        r = -1
        if len(a) >= 3:
            g = reduce(math.gcd, [b - a[0] for b in a[1:]])
            if g in (4, 6, 8, 12) and a[0] % g != 0:
                r = a[0] % g
            else:
                g = 0
        info[s] = (g, r, a[0] % 2)
    g = sets.map(lambda s: info[s][0]).to_numpy()
    r = sets.map(lambda s: info[s][1]).to_numpy()
    par = sets.map(lambda s: info[s][2]).to_numpy()
    t = df["target"].astype(int).to_numpy()
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    isex = (ex == t[:, None]).any(axis=1)
    y = df["rating"].astype(float).to_numpy()
    gs = np.where(g > 0, g, 1)
    ok = (g > 0) & ~isex & (t % 2 == par)
    cong = ok & (t % gs == r)
    non = ok & (t % gs != r)
    if cong.sum() == 0 or non.sum() == 0:
        return 0.0
    return float(y[cong].mean() - y[non].mean())
