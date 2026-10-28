# name: set_mean_rate_sd
# description: Standard deviation across number_sets of each set's overall yes-rate (sets with 9+ responses); observed above null means sets differ in overall liberality more than the model predicts (set-specific broad/narrow construal it misses), below means less.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 9]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
