# name: n1_small_target_gap
# description: For one-example sets, yes rate on targets 1-10 minus targets 11-100; observed < null means people endorse very small targets from a lone example less than the model predicts (a magnitude/plausibility prior the model lacks), observed > null the reverse.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = (s.str.count(",") == 0).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    a = one & (t <= 10)
    b = one & (t > 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
