# name: multi_set_size_trend
# description: Yes rate on 2-example sets minus yes rate on 4-example sets; observed > null means people narrow their generalization with more examples more than the model does (model under-produces the size-principle tightening), observed < null the reverse.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",").values + 1
    r = df["rating"].astype(float).values
    a = n == 2
    b = n == 4
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
