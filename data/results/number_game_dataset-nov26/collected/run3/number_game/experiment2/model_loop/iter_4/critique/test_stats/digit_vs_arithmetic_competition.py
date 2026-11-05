# name: digit_vs_arithmetic_competition
# description: For 2+ example sets whose examples all share a last digit d AND are all multiples of some k in 3..12 (e.g. 69,9,39,21; 28,98; 64,44), yes rate on non-example targets ending in d but not a multiple of k minus yes rate on targets that are multiples of k but do not end in d; observed above null means people favour the digit rule over the arithmetic rule more than the model does.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        if len(ex) < 2 or len(set(e % 10 for e in ex)) != 1:
            continue
        ks = [k for k in range(3, 13) if k != 5 and k != 10 and all(e % k == 0 for e in ex)]
        if ks:
            info[s] = (ex[0] % 10, max(ks), set(ex))
    if not info:
        return 0.0
    keep = sets.isin(list(info.keys())).to_numpy()
    ss = sets[keep].to_numpy()
    t = df["target"].astype(int).to_numpy()[keep]
    r = df["rating"].to_numpy().astype(float)[keep]
    d = np.array([info[a][0] for a in ss]); k = np.array([info[a][1] for a in ss])
    isex = np.array([b in info[a][2] for a, b in zip(ss, t)], dtype=bool)
    dig = (t % 10 == d) & ~isex; ari = (t % k == 0) & ~isex
    m1, m2 = dig & ~ari, ari & ~dig
    if m1.sum() == 0 or m2.sum() == 0:
        return 0.0
    return float(r[m1].mean() - r[m2].mean())
