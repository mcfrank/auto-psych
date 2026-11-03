# name: digit_sum_set_match_contrast
# description: Among multi-example sets whose examples all share one digit sum, yes rate for targets with that same digit sum minus yes rate for targets with a different digit sum; observed below null means the model over-produces digit-sum generalisation (people do not favour same-digit-sum targets as much as predicted), above means people use digit-sum rules more than the model.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = pd.unique(sets)
    codes = pd.Categorical(sets, categories=uniq).codes
    ok = np.zeros(len(uniq), bool); sval = np.zeros(len(uniq))
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")]
        d = {sum(map(int, str(e))) for e in v}
        if len(v) > 1 and len(d) == 1:
            ok[i] = True; sval[i] = d.pop()
    dsum = np.array([0] + [sum(map(int, str(k))) for k in range(1, 101)], float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    m = ok[codes]
    same = dsum[t] == sval[codes]
    a, b = m & same, m & ~same
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
