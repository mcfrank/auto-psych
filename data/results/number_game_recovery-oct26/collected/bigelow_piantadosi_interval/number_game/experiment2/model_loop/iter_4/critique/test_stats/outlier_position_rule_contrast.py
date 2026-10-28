# name: outlier_position_rule_contrast
# description: In 3-4 example sets with one ones-digit outlier, yes rate to non-example majority-digit targets when the outlier is in the second half of the presented order minus when it is in the first half; the volatile model predicts a late outlier hurts the rule more (negative), so observed above the null means people show less recency (order matters less) than the model, below means more.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    late, dg = {}, {}
    for s in sets.unique():
        e = [int(x) for x in s.split(",")]
        late[s], dg[s] = -1, -1
        if len(e) >= 3:
            vals, cnt = np.unique([x % 10 for x in e], return_counts=True)
            if cnt.max() == len(e) - 1:
                mm = int(vals[cnt.argmax()])
                pos = [i for i, x in enumerate(e) if x % 10 != mm][0]
                late[s] = int(pos >= len(e) / 2.0)
                dg[s] = mm
    lt = sets.map(late).to_numpy()
    m = sets.map(dg).to_numpy()
    t = df["target"].astype(int).to_numpy()
    is_ex = np.array([str(x) in s.split(",") for x, s in zip(t, sets)])
    r = df["rating"].to_numpy().astype(float)
    rule = (t % 10 == m) & ~is_ex & (lt >= 0)
    a, b = rule & (lt == 1), rule & (lt == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
