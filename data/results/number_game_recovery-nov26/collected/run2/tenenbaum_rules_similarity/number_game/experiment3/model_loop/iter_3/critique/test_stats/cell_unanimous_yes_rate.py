# name: cell_unanimous_yes_rate
# description: Share of (set, target) cells (with >=5 responses) in which every response is "yes"; observed above null means the model under-produces confident, unanimous endorsements (its yes-probabilities for clear members are too far below 1), below null means it over-produces them.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["size", "sum"])
    g = g[g["size"] >= 5]
    return float((g["sum"] == g["size"]).mean())
