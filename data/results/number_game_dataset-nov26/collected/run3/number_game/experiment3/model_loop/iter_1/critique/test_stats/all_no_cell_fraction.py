# name: all_no_cell_fraction
# description: Fraction of (set, target) cells whose responses are all "no"; observed above null means the model's predictions are not extreme enough on the low end (people unanimously reject more targets than the model's floor allows).
def test_statistic(df):
    g = df.groupby(["number_set", "target"])["rating"].max()
    return float((g == 0).mean())
