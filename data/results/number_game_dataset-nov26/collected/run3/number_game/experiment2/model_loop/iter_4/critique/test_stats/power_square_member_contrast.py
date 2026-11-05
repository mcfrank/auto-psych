# name: power_square_member_contrast
# description: For 2+ example sets whose examples all lie in one of squares/cubes/powers of 2/powers of 3, yes rate on non-example targets in such a covering rule minus yes rate on targets in none; observed below null means the model over-predicts how sharply people pick out these mathematical rules.
def test_statistic(df):
    rules = [set(k * k for k in range(1, 11)), {1, 8, 27, 64}, {1, 2, 4, 8, 16, 32, 64}, {1, 3, 9, 27, 81}]
    sets = df["number_set"].astype(str)
    cover = {}
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        if len(ex) < 2:
            continue
        cov = [R for R in rules if all(e in R for e in ex)]
        if cov:
            mem = set().union(*cov)
            cover[s] = (np.array([(v in mem) for v in range(101)]), set(ex))
    if not cover:
        return 0.0
    keep = sets.isin(list(cover.keys())).to_numpy()
    ss = sets[keep].to_numpy()
    t = df["target"].astype(int).to_numpy()[keep]
    r = df["rating"].to_numpy().astype(float)[keep]
    mem = np.array([cover[a][0][b] for a, b in zip(ss, t)], dtype=bool)
    isex = np.array([b in cover[a][1] for a, b in zip(ss, t)], dtype=bool)
    m1, m0 = mem & ~isex, ~mem
    if m1.sum() == 0 or m0.sum() == 0:
        return 0.0
    return float(r[m1].mean() - r[m0].mean())
