# name: unanimous_no_cell_fraction
# description: Fraction of distinct (set, target) cells in which every response is "no"; observed > null means people reject clearly-out-of-concept targets more uniformly than the model's guessing/stray-example noise allows (observed < null: humans noisier than the model).
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    s = df["rating"].astype(float).groupby(key).max()
    return float((s == 0).mean())
