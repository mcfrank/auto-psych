# name: common_factor_rule_gap
# description: Among multi-example sets whose examples share a greatest common divisor g>=3, the yes rate for targets divisible by g minus the yes rate for targets not divisible by g; observed > model means people generalize a shared "multiples of g" rule more than pairwise exemplar divisibility/digit similarity predicts.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    g = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        gg = 0
        for x in ex:
            gg = math.gcd(gg, x)
        g[s] = gg if len(ex) >= 2 else 0
    gv = sets.map(g).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    m = gv >= 3
    if m.sum() == 0:
        return 0.0
    div = (t % np.where(gv > 0, gv, 1)) == 0
    a, b = m & div, m & ~div
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
