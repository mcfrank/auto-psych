# name: between_set_yes_rate_sd
# description: Standard deviation across number sets (with >= 18 responses) of each set's overall yes-rate; observed above the null means sets differ in how generously they are generalized more than the model predicts (its set-level breadth is too compressed), below means the model exaggerates set-to-set differences.
def test_statistic(df):
    g = df.groupby(df["number_set"].astype(str))["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 18]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
