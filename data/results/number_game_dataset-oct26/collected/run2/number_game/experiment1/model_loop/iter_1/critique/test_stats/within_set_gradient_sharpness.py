# name: within_set_gradient_sharpness
# description: Mean, over sets with >=4 distinct targets, of the variance of per-target yes rates within the set; observed > model means people's generalization across targets of the same set is more all-or-none (rule-like, extreme) than the model's graded similarity predicts, observed < model means flatter.
def test_statistic(df):
    g = df.assign(rating=df["rating"].astype(float)).groupby(["number_set", "target"])["rating"].mean().reset_index()
    cnt = g.groupby("number_set")["target"].transform("count")
    g = g[cnt >= 4]
    if len(g) == 0:
        return 0.0
    v = g.groupby("number_set")["rating"].var(ddof=0)
    return float(v.mean())
