# name: between_set_yes_rate_variance
# description: Variance across sets (sets with at least 10 trials) of each set's overall yes rate; observed above null means sets differ more in how broadly people generalise (some sets near all-yes, others near all-no) than the model predicts, below null means the model exaggerates set-to-set differences in breadth.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].var())
