# name: set_yes_rate_dispersion
# description: Standard deviation across number sets (with 10+ responses) of the per-set yes rate; observed > null means people's overall willingness to generalize varies across sets more than the model predicts (some sets drawing near-total rejection, others broad acceptance), observed < null means less.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "count"])
    g = g[g["count"] >= 10]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
