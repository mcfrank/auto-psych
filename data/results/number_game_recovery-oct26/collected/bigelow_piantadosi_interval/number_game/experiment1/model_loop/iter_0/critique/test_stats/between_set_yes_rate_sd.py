# name: between_set_yes_rate_sd
# description: Standard deviation across example sets (with >=10 trials) of each set's yes-rate; observed above null means generalization breadth varies more from set to set than the model predicts (model too homogeneous), below means less.
def test_statistic(df):
    g = df.groupby(df["number_set"].astype(str))["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    return float(g["mean"].std()) if len(g) > 1 else 0.0
