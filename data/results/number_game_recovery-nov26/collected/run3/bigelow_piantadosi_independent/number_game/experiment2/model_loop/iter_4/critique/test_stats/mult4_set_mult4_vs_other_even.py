# name: mult4_set_mult4_vs_other_even
# description: In sets of >=2 examples all divisible by 4, yes rate on non-example multiple-of-4 targets minus yes rate on even non-multiple-of-4 targets; observed < null means people treat such sets as "even"/ranges more than the model's "multiples of 4" reading predicts.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    feat = {}
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        feat[s] = (len(ex) >= 2 and all(v % 4 == 0 for v in ex), ex)
    ok = sets.map(lambda s: feat[s][0]).to_numpy(bool)
    isex = np.array([ti in feat[s][1] for s, ti in zip(sets, t)])
    base = ok & ~isex
    a = base & (t % 4 == 0); b = base & (t % 2 == 0) & (t % 4 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
