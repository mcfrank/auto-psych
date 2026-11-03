# name: set_yes_rate_sd
# description: Standard deviation across example sets (with >=20 answers) of the set's overall yes-rate; observed > null means people's breadth of generalisation differs between sets more than the model predicts, observed < null means the model exaggerates set-to-set differences in breadth.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
