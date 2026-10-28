# name: between_set_yes_rate_sd
# description: Standard deviation, across example sets with at least 20 responses, of the per-set yes-rate; observed above the null means the model makes sets too similar in overall generality (under-produces set-to-set differences in concept breadth), below means it exaggerates them.
def test_statistic(df):
    g = df.groupby(df["number_set"].astype(str))["rating"].agg(["mean", "count"])
    g = g[g["count"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
