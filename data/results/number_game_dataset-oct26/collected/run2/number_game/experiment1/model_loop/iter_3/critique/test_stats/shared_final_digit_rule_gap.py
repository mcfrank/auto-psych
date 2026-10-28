# name: shared_final_digit_rule_gap
# description: Among multi-example sets whose examples all end in the same digit, yes rate for targets with that final digit minus yes rate for other targets; observed > model means people apply an "ends in d" rule more sharply than coherence-stretched digit attention predicts (observed < model means the model over-weights digit coherence).
def test_statistic(df):
    s = df["number_set"].astype(str)
    dmap = {}
    for u in s.unique():
        v = [int(z) for z in u.split(",")]
        ds = set(z % 10 for z in v)
        dmap[u] = ds.pop() if (len(v) >= 2 and len(ds) == 1) else -1
    d = s.map(dmap).values.astype(int)
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    m = d >= 0
    a = m & (t % 10 == d)
    b = m & (t % 10 != d)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
