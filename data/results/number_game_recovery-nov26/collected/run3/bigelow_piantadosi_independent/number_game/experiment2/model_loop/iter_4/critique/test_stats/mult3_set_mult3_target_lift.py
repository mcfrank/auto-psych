# name: mult3_set_mult3_target_lift
# description: In sets of >=2 examples all divisible by 3 (and not all divisible by 5), yes rate on non-example multiple-of-3 targets minus yes rate on non-multiple-of-3 targets; observed < null means the model over-generalises "multiples of 3".
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    feat = {}
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        feat[s] = (len(ex) >= 2 and all(v % 3 == 0 for v in ex) and not all(v % 5 == 0 for v in ex), ex)
    ok = sets.map(lambda s: feat[s][0]).to_numpy(bool)
    isex = np.array([ti in feat[s][1] for s, ti in zip(sets, t)])
    base = ok & ~isex
    a = base & (t % 3 == 0); b = base & (t % 3 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
