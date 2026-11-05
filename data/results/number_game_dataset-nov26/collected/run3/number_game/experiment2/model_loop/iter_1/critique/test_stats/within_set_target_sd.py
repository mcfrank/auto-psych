# name: within_set_target_sd
# description: Mean over sets with >=5 distinct targets of the SD of per-target yes rates within the set; observed above null means people discriminate between targets of the same set more sharply than the model (below = model's gradients are too sharp / all-or-none).
def test_statistic(df):
    c = df.groupby(["number_set", "target"])["rating"].mean().reset_index()
    g = c.groupby("number_set")["rating"]
    sd = g.std(ddof=0)[g.size() >= 5]
    return float(sd.mean()) if len(sd) else 0.0
