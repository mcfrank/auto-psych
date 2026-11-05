# name: single_example_cell_polarization
# description: Among (set, target) cells from one-example sets, fraction of cells whose yes-rate is <=1/9 or >=8/9; observed > null means people answer all-or-none after a lone example even though the model (no coincidence, so its least-steep slope) predicts graded answers there; observed < null means the model is too polarized there.
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    d = df[n == 1]
    y = d.groupby(["number_set", "target"])["rating"].mean()
    if len(y) == 0:
        return 0.0
    return float(((y <= 1/9 + 1e-9) | (y >= 8/9 - 1e-9)).mean())
