# name: set_level_yes_dispersion
# description: Standard deviation across number_sets (with >=10 trials) of each set's overall yes rate; observed above null means people differentiate sets more strongly (some sets accepted almost everywhere, others almost nowhere) than the model, below means the model over-differentiates sets.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    return float(g["mean"].std()) if len(g) > 1 else 0.0
