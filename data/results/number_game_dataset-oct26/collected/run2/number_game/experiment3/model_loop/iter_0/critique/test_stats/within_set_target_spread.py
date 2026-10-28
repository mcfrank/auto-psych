# name: within_set_target_spread
# description: For sets asked with >=8 distinct targets, the mean over sets of the SD of per-target yes-rates across targets; observed > null means people discriminate between targets of the same set more sharply than the model (model too smooth), observed < null means the model is too sharp.
def test_statistic(df):
    g = df.groupby(["number_set", "target"], sort=False)["rating"].mean().reset_index()
    sd = g.groupby("number_set")["rating"].agg(["std", "size"])
    sd = sd[sd["size"] >= 8]
    if len(sd) == 0:
        return 0.0
    return float(sd["std"].mean())
