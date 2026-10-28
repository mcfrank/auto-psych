# name: size_principle_far_target_k_slope
# description: Among "far" targets (>=15 from every example, final digit shared with no example, and coprime with every example), least-squares slope of yes-rate on number of examples k; observed more negative than the null means people tighten generalization with more examples (size principle) more than the averaged-similarity model does, more positive means less.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    exm = {s: np.array([int(v) for v in s.split(",")] + [-1000] * (4 - len(s.split(",")))) for s in sets}
    E = np.stack(ns.map(exm).values)[:, :4]
    k = (E > 0).sum(1)
    t = df["target"].astype(int).values
    valid = E > 0
    dist = np.where(valid, np.abs(E - t[:, None]), 1000)
    far = dist.min(1) >= 15
    digm = (valid & ((E % 10) == (t[:, None] % 10))).any(1)
    Ep = np.where(valid, E, 1)
    g = np.gcd(Ep, t[:, None])
    share = (valid & (g > 1)).any(1)
    sel = far & ~digm & ~share
    y = df["rating"].astype(float).values
    if sel.sum() < 3 or np.unique(k[sel]).size < 2:
        return 0.0
    kk = k[sel].astype(float)
    yy = y[sel]
    return float(np.polyfit(kk, yy, 1)[0])
