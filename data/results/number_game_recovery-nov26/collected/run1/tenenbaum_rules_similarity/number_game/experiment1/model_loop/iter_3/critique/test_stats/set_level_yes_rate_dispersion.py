# name: set_level_yes_rate_dispersion
# description: Standard deviation across sets (with >=20 responses) of each set's overall yes-rate; observed > null means people's overall willingness to generalize differs between sets more than the model predicts (model too homogeneous across sets, e.g. too broad for tight sets or too narrow for loose ones), observed < null the reverse.
def test_statistic(df):
    g = df["rating"].astype(float).groupby(df["number_set"].astype(str)).agg(["mean", "size"])
    g = g[g["size"] >= 20]
    return float(g["mean"].std(ddof=0)) if len(g) > 1 else 0.0
