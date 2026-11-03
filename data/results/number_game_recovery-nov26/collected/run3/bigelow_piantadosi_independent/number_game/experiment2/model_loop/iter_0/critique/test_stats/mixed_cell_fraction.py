# name: mixed_cell_fraction
# description: Fraction of (number_set, target) cells whose repeated answers are not unanimous (some yes and some no); observed above null means the model's p_yes is too extreme (people are less all-or-none than the near-hard commitment predicts), and below null means it is too graded.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].agg(["sum", "size"])
    return float(((g["sum"] > 0) & (g["sum"] < g["size"])).mean())
