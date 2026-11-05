# name: multi_within_set_target_variance
# description: Mean, over multi-example sets probed with at least 4 targets, of the variance of per-target yes rates within the set; observed > null means people discriminate between targets of the same set more sharply than the model (observed < null: model over-discriminates).
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",").values + 1
    d = df.loc[n >= 2, ["number_set", "target", "rating"]]
    pm_ = d.groupby(["number_set", "target"])["rating"].mean().reset_index()
    g = pm_.groupby("number_set")["rating"]
    v = g.var(ddof=0)
    c = g.size()
    v = v[c >= 4]
    return float(v.mean()) if len(v) else 0.0
