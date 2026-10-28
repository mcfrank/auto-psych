# name: set_level_yes_rate_dispersion
# description: Standard deviation across example sets (with 40+ rows) of each set's overall yes-rate; observed > null means people differentiate sets (some sets read as broad, others as narrow) more than the model does, observed < null means the model over-differentiates.
def test_statistic(df):
    y = df["rating"].astype(float)
    g = y.groupby(df["number_set"].astype(str)).agg(["mean", "size"])
    g = g[g["size"] >= 40]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
