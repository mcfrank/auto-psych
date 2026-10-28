# name: outlier_cost_to_rule
# description: Yes rate to non-example targets sharing the majority ones digit in clean sets (3-4 examples all sharing a ones digit) minus the same rate in outlier sets (all but one example share it); observed above the null means one contradicting example hurts rule-following more for people than for the model, below means people are more robust to the outlier than the model.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    kind, dg = {}, {}
    for s in sets.unique():
        e = [int(x) for x in s.split(",")]
        kind[s], dg[s] = 0, -1
        if len(e) >= 3:
            vals, cnt = np.unique([x % 10 for x in e], return_counts=True)
            if cnt.max() == len(e):
                kind[s], dg[s] = 1, int(vals[0])
            elif cnt.max() == len(e) - 1:
                kind[s], dg[s] = 2, int(vals[cnt.argmax()])
    k = sets.map(kind).to_numpy()
    m = sets.map(dg).to_numpy()
    t = df["target"].astype(int).to_numpy()
    is_ex = np.array([str(x) in s.split(",") for x, s in zip(t, sets)])
    r = df["rating"].to_numpy().astype(float)
    rule = (t % 10 == m) & ~is_ex
    a, b = rule & (k == 1), rule & (k == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
