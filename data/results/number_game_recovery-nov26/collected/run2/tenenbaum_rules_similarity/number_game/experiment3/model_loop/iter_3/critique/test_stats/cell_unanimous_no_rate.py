# name: cell_unanimous_no_rate
# description: Share of (set, target) cells (with >=5 responses) in which every response is "no"; observed above null means people reject more cells with total unanimity than the model's graded probabilities allow (the model spreads too much yes-probability over cells that people treat as clear non-members), below null the reverse.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["size", "sum"])
    g = g[g["size"] >= 5]
    return float((g["sum"] == 0).mean())
