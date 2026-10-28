# name: set_level_yes_rate_dispersion
# description: Standard deviation across example sets with 40+ rows of each set's overall yes-rate; observed > null means people treat some sets as broad and others as narrow more strongly than the model does (model compresses set-level differences), observed < null the reverse.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 40]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
