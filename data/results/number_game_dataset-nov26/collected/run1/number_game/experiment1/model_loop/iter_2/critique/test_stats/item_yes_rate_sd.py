# name: item_yes_rate_sd
# description: SD across (set,target) items (with >=3 responses) of the item yes-rate; observed > null means responses are more polarized/item-consistent than the model predicts (model too graded/noisy), observed < null the reverse.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "count"])
    g = g[g["count"] >= 3]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
