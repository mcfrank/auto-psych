# name: multi_small_target_gap
# description: Over multi-example sets, yes rate on targets 1-10 minus yes rate on targets 11-100; observed < null means the model over-endorses small targets (which belong to many concepts such as squares, powers, small multiples) relative to people, observed > null means it under-endorses them.
def test_statistic(df):
    s = df["number_set"].astype(str)
    multi = (s.str.count(",") > 0).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    a = multi & (t <= 10)
    b = multi & (t > 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
