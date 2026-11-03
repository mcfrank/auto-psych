# name: set_level_yes_rate_sd
# description: Standard deviation across example sets (with >=20 trials) of each set's overall yes rate; observed > null means people's breadth of generalisation differs more between sets (more all-or-none per set) than the model predicts, observed < null means the model's set-to-set contrasts are too extreme.
def test_statistic(df):
    g = df.groupby(df["number_set"].astype(str))["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
