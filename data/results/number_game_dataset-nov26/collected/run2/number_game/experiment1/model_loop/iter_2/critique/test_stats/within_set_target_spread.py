# name: within_set_target_spread
# description: Mean, over sets asked with >=8 distinct targets, of the standard deviation of per-target yes-rates within the set; observed > null means people discriminate among targets of the same set more sharply (more all-or-none) than the model's graded overlap predicts, observed < null means the model over-discriminates.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].mean().reset_index()
    cnt = g.groupby("number_set")["target"].transform("size")
    g = g[cnt >= 8]
    if len(g) == 0:
        return 0.0
    sd = g.groupby("number_set")["rating"].std(ddof=0)
    return float(sd.mean())
