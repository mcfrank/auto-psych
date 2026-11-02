# name: within_set_target_profile_sd
# description: Mean over sets with >=100 trials of the SD across targets (each with >=5 answers) of target-level yes rates; observed above null means people's generalization profiles within a set are sharper (more all-or-none across targets) than the model's, below means flatter.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= 5]
    sz = df.groupby("number_set").size()
    big = sz[sz >= 100].index
    g = g[g["number_set"].isin(big)]
    sds = g.groupby("number_set")["mean"].std().dropna()
    if len(sds) == 0:
        return 0.0
    return float(sds.mean())
