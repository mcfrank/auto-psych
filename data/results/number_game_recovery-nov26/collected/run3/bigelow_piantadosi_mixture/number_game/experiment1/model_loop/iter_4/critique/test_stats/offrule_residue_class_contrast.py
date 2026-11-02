# name: offrule_residue_class_contrast
# description: Among sets of 3+ examples whose pairwise differences share a gcd d >= 4 (d != 10) but which are not multiples of d (e.g. 33,17,5,9 = 1 mod 4), yes rate for non-example targets in the same residue class mod d minus targets outside it; observed > null means people detect "same remainder" regularities the model has no concept for.
def test_statistic(df):
    sets = df["number_set"].unique()
    info = {}
    for s in sets:
        l = [int(x) for x in s.split(",")]
        if len(l) < 3:
            continue
        g = 0
        for a in l[1:]:
            g = math.gcd(g, abs(a - l[0]))
        if g >= 4 and g != 10 and l[0] % g != 0:
            info[s] = (g, l[0] % g, set(l))
    if not info:
        return 0.0
    sub = df[df["number_set"].isin(list(info))]
    items = sub.groupby(["number_set", "target"])["rating"].mean().reset_index()
    inres = np.array([t % info[s][0] == info[s][1] for s, t in zip(items["number_set"], items["target"])])
    isex = np.array([t in info[s][2] for s, t in zip(items["number_set"], items["target"])])
    a = items["rating"][inres & ~isex]
    b = items["rating"][~inres & ~isex]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
