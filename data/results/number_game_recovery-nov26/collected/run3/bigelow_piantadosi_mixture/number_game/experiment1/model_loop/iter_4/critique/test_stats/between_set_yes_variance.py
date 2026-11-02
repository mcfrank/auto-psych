# name: between_set_yes_variance
# description: Variance across sets (with >= 10 trials) of the set-level yes rate; observed > null means people's breadth of generalisation differs between sets more than the model predicts (some sets far broader/narrower), observed < null means the model makes set breadths too heterogeneous.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].var())
