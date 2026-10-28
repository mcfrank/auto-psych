# name: set_yes_rate_sd
# description: Standard deviation across sets (with >=20 responses) of each set's overall yes-rate; observed above null means sets differ in how broadly people generalize more than the model predicts (model too uniform in breadth across sets), below null less.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].astype(float).std())
