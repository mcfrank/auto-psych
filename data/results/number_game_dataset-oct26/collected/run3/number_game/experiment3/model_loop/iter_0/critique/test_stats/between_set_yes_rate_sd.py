# name: between_set_yes_rate_sd
# description: Standard deviation across distinct sets of the per-set yes-rate (sets with >= 20 responses); observed > null means human generalization breadth varies more across sets than the model predicts (some sets far broader/narrower), observed < null means the model over-differentiates sets.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
