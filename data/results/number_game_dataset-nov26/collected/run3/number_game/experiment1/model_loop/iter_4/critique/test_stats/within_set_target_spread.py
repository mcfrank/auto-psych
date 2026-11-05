# name: within_set_target_spread
# description: Mean, over sets tested with >=8 distinct targets, of the standard deviation of per-target yes rates within the set; observed below null means people discriminate among a set's targets less sharply than the model (flatter generalization), above null means more sharply (more all-or-none).
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].mean().reset_index()
    cnt = g.groupby("number_set")["target"].transform("count")
    g = g[cnt >= 8]
    if len(g) == 0:
        return 0.0
    return float(g.groupby("number_set")["rating"].std(ddof=0).mean())
