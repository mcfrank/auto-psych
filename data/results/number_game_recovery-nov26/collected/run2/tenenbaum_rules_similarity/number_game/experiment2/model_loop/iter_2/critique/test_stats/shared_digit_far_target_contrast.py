# name: shared_digit_far_target_contrast
# description: For non-example targets more than 15 away from every example, yes rate when the target shares at least one decimal digit with some example minus yes rate when it shares none; observed above the null means people use loose digit similarity beyond what the model's all-examples-contain-digit rules produce (below = less digit-driven than the model).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).to_numpy()
    uniq = sets.unique()
    ex_map = {s: np.array([int(x) for x in s.split(",")]) for s in uniq}
    mind = np.empty(len(df)); share = np.zeros(len(df), dtype=bool)
    ss = sets.to_numpy()
    tdig = [set(str(v)) for v in range(101)]
    for s in uniq:
        idx = np.where(ss == s)[0]
        ex = ex_map[s]
        tt = t[idx]
        mind[idx] = np.abs(tt[:, None] - ex[None, :]).min(axis=1)
        digs = set("".join(str(e) for e in ex))
        lut = np.array([len(tdig[v] & digs) > 0 for v in range(101)])
        share[idx] = lut[tt]
    far = mind > 15
    r = df["rating"].to_numpy()
    a, b = far & share, far & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
