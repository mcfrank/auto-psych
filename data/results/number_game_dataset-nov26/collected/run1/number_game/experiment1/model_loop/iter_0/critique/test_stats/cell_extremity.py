# name: cell_extremity
# description: Fraction of (set, target) cells whose yes-rate is <= 0.12 or >= 0.88 (near-unanimous no or yes); observed below null_mean means the model's predictions are too all-or-none (people are more graded/uncertain), above means people are more categorical than the model.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].mean().values
    if len(g) == 0:
        return 0.0
    return float(np.mean((g <= 0.12) | (g >= 0.88)))
