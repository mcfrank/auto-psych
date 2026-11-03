# name: narrow_vs_wide_range_inside_gap
# description: For sets of 2+ distinct examples, yes rate on non-example targets strictly inside [min, max] that share no final digit with any example, for sets with range <= 25 minus sets with range >= 60; observed above null means people fill in narrow clusters more (or wide spans less) than the model's interval/run account predicts.
def test_statistic(df):
    info = {}
    for s in df["number_set"].astype(str).unique():
        ex = [int(x) for x in s.split(",")]
        info[s] = (len(set(ex)), min(ex), max(ex), set(x % 10 for x in ex), set(ex))
    ns = df["number_set"].astype(str).values
    tg = df["target"].astype(int).values
    k = np.array([info[s][0] for s in ns])
    mn = np.array([info[s][1] for s in ns])
    mx = np.array([info[s][2] for s in ns])
    ok = np.array([(t % 10) not in info[s][3] and t not in info[s][4] for s, t in zip(ns, tg)])
    inside = (k >= 2) & (tg > mn) & (tg < mx) & ok
    rng = mx - mn
    r = df["rating"].values.astype(float)
    a = inside & (rng <= 25)
    b = inside & (rng >= 60)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
