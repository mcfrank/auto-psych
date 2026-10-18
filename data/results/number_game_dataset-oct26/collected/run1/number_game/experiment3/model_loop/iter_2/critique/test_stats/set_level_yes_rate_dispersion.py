# name: set_level_yes_rate_dispersion
# description: Standard deviation across sets (with >=30 responses) of each set's overall yes rate; observed > null means sets differ in how broadly people generalise more than the model's features capture (model too homogeneous across sets), observed < null the reverse.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "count"])
    g = g[g["count"] >= 30]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
