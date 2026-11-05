# name: item_mean_variance
# description: Variance across (set, target) pairs of the pair's mean yes rate; observed > null means people differentiate items more sharply (more all-yes / all-no pairs) than the model, observed < null means the model is over-confident.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].mean()
    return float(g.var())
