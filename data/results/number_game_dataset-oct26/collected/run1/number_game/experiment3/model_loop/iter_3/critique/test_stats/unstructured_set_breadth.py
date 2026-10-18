# name: unstructured_set_breadth
# description: Yes-rate on targets sharing no final digit with any example in "unstructured" sets (>=3 examples, gcd 1, mixed parity, >=3 distinct final digits) minus the same rate in singleton sets; observed > null means people generalise more broadly from diverse, rule-less examples ("any number") than the model predicts, observed < null that they generalise less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    cats = pd.unique(sets)
    info = {}
    for s in cats:
        xs = [int(x) for x in s.split(",")]
        g = 0
        for x in xs:
            g = math.gcd(g, x)
        par = {x % 2 for x in xs}
        units = {x % 10 for x in xs}
        kind = 1 if len(xs) == 1 else (2 if (len(xs) >= 3 and g == 1 and len(par) == 2 and len(units) >= 3) else 0)
        info[s] = (kind, units)
    sv = sets.values
    t = df["target"].astype(int).values
    kind = np.array([info[s][0] for s in sv])
    dm = np.array([(ti % 10) in info[s][1] for s, ti in zip(sv, t)])
    r = df["rating"].values.astype(float)
    a = (kind == 2) & ~dm
    b = (kind == 1) & ~dm
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
