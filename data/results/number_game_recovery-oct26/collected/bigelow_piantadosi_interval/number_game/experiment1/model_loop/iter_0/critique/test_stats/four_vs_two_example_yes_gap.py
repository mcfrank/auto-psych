# name: four_vs_two_example_yes_gap
# description: Yes-rate on 4-example sets minus yes-rate on 2-example sets; observed more negative than null means people tighten generalization with more examples more than the model does, more positive means less.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",").to_numpy() + 1
    r = df["rating"].astype(float).to_numpy()
    a, b = r[n == 4], r[n == 2]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
