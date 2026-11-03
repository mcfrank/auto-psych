# name: between_set_yes_rate_variance
# description: Variance across example sets (with >=10 trials) of the per-set yes rate; observed above null means people's generalisation breadth differs between sets more sharply than the model predicts (observed below null: the model makes sets too different).
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].var())
