# name: set_breadth_spread
# description: Standard deviation across example sets (with >=20 responses) of each set's mean yes rate; observed above null means sets differ in how broadly people generalise more than the model predicts (some sets near-zero, others very broad), below null means less.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
