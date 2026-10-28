# name: between_set_yes_rate_sd
# description: Standard deviation across sets (with at least 20 trials) of each set's overall yes rate; observed above null means people differ more between sets in overall willingness to generalize than the model predicts (model too homogeneous), below null means less.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    return float(g["mean"].std()) if len(g) > 1 else 0.0
