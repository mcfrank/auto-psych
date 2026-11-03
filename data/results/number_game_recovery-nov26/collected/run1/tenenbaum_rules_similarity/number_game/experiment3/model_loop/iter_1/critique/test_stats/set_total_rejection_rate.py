# name: set_total_rejection_rate
# description: Fraction of distinct sets (with 10+ responses) on which no target ever received a yes; observed above null means people reject all probes of some sets wholesale more often than the model predicts, below means the model predicts more all-no sets than observed.
def test_statistic(df):
    g = df["rating"].astype(float).groupby(df["number_set"].astype(str))
    cnt = g.size()
    s = g.sum()[cnt >= 10]
    return float((s == 0).mean()) if len(s) else 0.0
