# name: between_set_rate_dispersion
# description: Standard deviation across sets (with at least 20 trials) of the per-set yes rate; observed above null means people's overall willingness to say yes varies between sets more than the model's concept posterior produces.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
