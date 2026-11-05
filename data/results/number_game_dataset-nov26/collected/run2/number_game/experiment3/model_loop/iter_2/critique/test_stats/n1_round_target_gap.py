# name: n1_round_target_gap
# description: For one-example sets, yes rate on targets that are multiples of 10 minus other targets; observed < null means people reject round targets from a lone example more than the model predicts (observed > null: model under-produces a round-target bonus).
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = (s.str.count(",") == 0).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    a = one & (t % 10 == 0)
    b = one & (t % 10 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
