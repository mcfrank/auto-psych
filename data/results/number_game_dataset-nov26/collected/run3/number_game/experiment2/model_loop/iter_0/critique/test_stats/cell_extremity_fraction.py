# name: cell_extremity_fraction
# description: Fraction of (set, target) cells whose mean yes rate is <= 0.15 or >= 0.85; observed above null means human judgments are more all-or-none (the model's predictions are too graded/intermediate), below null means they are more graded than the model.
def test_statistic(df):
    m = df.groupby(["number_set", "target"])["rating"].mean().to_numpy()
    if len(m) == 0:
        return 0.0
    return float(np.mean((m <= 0.15) | (m >= 0.85)))
