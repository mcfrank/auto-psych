# name: unanimous_cell_fraction
# description: Fraction of (number_set, target) cells answered by exactly 5 participants in which all 5 gave the same answer (all yes or all no); observed above null means human generalisation is more categorical/consensual (cell probabilities pushed to 0/1) than the model's graded predictions, below means people split more than predicted.
def test_statistic(df):
    cell = (df["number_set"].astype(str) + "|" + df["target"].astype(str)).to_numpy()
    g = df["rating"].astype(float).groupby(cell).agg(["sum", "count"])
    g = g[g["count"] == 5]
    return float(((g["sum"] == 0) | (g["sum"] == 5)).mean())
