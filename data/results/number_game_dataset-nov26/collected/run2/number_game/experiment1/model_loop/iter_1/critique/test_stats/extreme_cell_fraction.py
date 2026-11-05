# name: extreme_cell_fraction
# description: Fraction of (set,target) cells whose mean yes-rate is <=0.15 or >=0.85; observed < null means human responses are less categorical (more graded/noisy) than the model predicts, observed > null means more consensus than the model.
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].mean().values
    if len(g) == 0:
        return 0.0
    return float(np.mean((g <= 0.15) | (g >= 0.85)))
