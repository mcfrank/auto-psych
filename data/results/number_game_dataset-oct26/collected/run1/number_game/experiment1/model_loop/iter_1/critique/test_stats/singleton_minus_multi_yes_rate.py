# name: singleton_minus_multi_yes_rate
# description: Yes rate on 1-example sets minus yes rate on sets with 2-4 examples; summed similarity predicts more examples -> more yes (negative value), so an observed value above the null means the model under-produces generalization from single examples relative to multi-example sets (no size-principle tightening).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    r = df["rating"].astype(float)
    a = r[n == 1]
    b = r[n >= 2]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
