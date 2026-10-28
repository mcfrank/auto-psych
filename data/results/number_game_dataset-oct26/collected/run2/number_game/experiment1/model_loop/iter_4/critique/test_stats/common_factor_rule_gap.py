# name: common_factor_rule_gap
# description: Among multi-example sets whose examples all share a greatest common divisor g>=3, yes rate for targets divisible by g minus yes rate for targets not divisible by g; observed > model means people generalize the shared "multiples of g" rule more sharply than graded-factor coherence-tuned similarity predicts (tracks whether the iter-3 refinement closed this gap).
def test_statistic(df):
    s = df["number_set"].astype(str)
    uniq = s.unique()
    gmap = {}
    for u in uniq:
        v = [int(z) for z in u.split(",")]
        g = 0
        for z in v:
            g = math.gcd(g, z)
        gmap[u] = g if (len(v) >= 2 and g >= 3) else 0
    g = s.map(gmap).values.astype(int)
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    m = g > 0
    if m.sum() == 0:
        return 0.0
    div = np.zeros(len(t), bool)
    div[m] = (t[m] % g[m]) == 0
    a = m & div
    b = m & ~div
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
