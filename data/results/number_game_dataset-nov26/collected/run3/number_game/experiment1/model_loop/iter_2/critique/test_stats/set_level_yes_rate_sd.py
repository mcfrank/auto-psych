# name: set_level_yes_rate_sd
# description: Standard deviation across example sets (sets with >=18 trials) of each set's overall yes rate; observed above null means people's overall willingness to generalize differs between sets more than the model predicts, below null means less.
def test_statistic(df):
    g = df.assign(rr=df["rating"].astype(float)).groupby(df["number_set"].astype(str))["rr"].agg(["mean", "count"])
    g = g[g["count"] >= 18]
    return float(g["mean"].std()) if len(g) > 1 else 0.0
