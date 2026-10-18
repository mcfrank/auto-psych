# name: set_mean_variance
# description: Variance across number sets (with >=18 responses) of the per-set mean yes rate; observed above null means sets differ in overall generalization breadth more than the digit-similarity model with a single intercept allows.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 18]
    return float(g["mean"].var(ddof=0)) if len(g) > 1 else 0.0
