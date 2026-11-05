# name: outlier_decade_generalization
# description: For 3-4 example sets where all but one example share a last digit, yes rate on non-example targets in the same decade (tens digit) as the odd-one-out example; observed above null means people generalize around a rule-violating example more than the model's strict likelihood allows (below null: less).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        if len(ex) < 3:
            continue
        ld = [e % 10 for e in ex]
        for i, e in enumerate(ex):
            rest = ld[:i] + ld[i + 1:]
            if len(set(rest)) == 1 and ld[i] != rest[0]:
                info[s] = (e // 10 if e < 100 else 9, tuple(ex))
                break
    if not info:
        return 0.0
    keep = sets.isin(list(info.keys())).to_numpy()
    sub_s = sets[keep].to_numpy()
    t = df["target"].astype(int).to_numpy()[keep]
    r = df["rating"].to_numpy().astype(float)[keep]
    dec = np.array([info[s][0] for s in sub_s])
    isex = np.array([tt in info[s][1] for s, tt in zip(sub_s, t)]) if len(t) else np.zeros(0, bool)
    m = ((np.minimum(t, 99) // 10) == dec) & (~isex)
    return float(r[m].mean()) if m.sum() > 0 else 0.0
