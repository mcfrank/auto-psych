# name: gcd_rule_member_contrast
# description: Among sets of 2+ examples whose gcd is >= 3, yes rate for unseen targets divisible by the gcd minus unseen targets not divisible by it; observed > null means people generalise along the shared-multiple rule more sharply than the model, observed < null means the model over-commits to multiples rules.
def test_statistic(df):
    sets = df["number_set"].unique()
    gm = {}
    for s in sets:
        v = [int(x) for x in str(s).split(",")]
        if len(v) >= 2:
            g = 0
            for x in v:
                g = math.gcd(g, x)
            if g >= 3:
                gm[s] = g
    ns = df["number_set"]
    g = ns.map(gm)
    keep = g.notna().values
    if keep.sum() == 0:
        return 0.0
    t = df["target"].astype(int).values
    padded = ("," + ns.astype(str) + ",").values
    tstr = ("," + df["target"].astype(int).astype(str) + ",").values
    member = np.array([b.find(a) >= 0 for a, b in zip(tstr, padded)])
    gv = np.where(keep, g.fillna(1).values, 1).astype(int)
    div = (t % gv) == 0
    r = df["rating"].astype(float).values
    a = keep & ~member & div
    b = keep & ~member & ~div
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
