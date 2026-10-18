# name: within_set_target_spread
# description: Mean over sets with >=4 distinct targets (each with >=5 responses) of the standard deviation of per-target yes-rates within the set; observed > null means people discriminate among targets of the same set more sharply than the model predicts (model's target-level gradients too flat), observed < null the reverse.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= 5]
    s = g.groupby("number_set")["mean"].agg(["std", "count"])
    s = s[s["count"] >= 4]
    if len(s) == 0:
        return 0.0
    return float(s["std"].mean())
