# name: two_example_dip
# description: Yes-rate on 2-example sets minus the mean yes-rate on 3- and 4-example sets; observed < null means people are more conservative on two-example sets (relative to larger sets) than the model's monotone capacity limit predicts.
def test_statistic(df):
    n = df["number_set"].str.count(",").to_numpy() + 1
    r = df["rating"].astype(float).to_numpy()
    a = r[n == 2]; b = r[n >= 3]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
