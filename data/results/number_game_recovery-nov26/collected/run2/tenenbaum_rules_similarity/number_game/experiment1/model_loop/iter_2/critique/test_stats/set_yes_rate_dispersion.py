# name: set_yes_rate_dispersion
# description: Standard deviation across number_sets (those with >=20 trials) of the per-set yes-rate; observed above null means people differ between sets more than the model (model's set-level generalisation too uniform), observed below null means the model makes sets too different (people are more uniform across sets).
def test_statistic(df):
    g = df.groupby(df["number_set"].astype(str))["rating"].agg(["mean", "count"])
    g = g[g["count"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std(ddof=0))
