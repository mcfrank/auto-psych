# name: item_mean_variance
# description: Variance across (set, target) items of the item mean yes rate; observed above null means human judgments differ across items more sharply than the digit-similarity model predicts (model too flat / missing item-relevant features), below means the model over-discriminates.
def test_statistic(df):
    m = df.groupby(["number_set", "target"])["rating"].mean()
    return float(m.var(ddof=0))
