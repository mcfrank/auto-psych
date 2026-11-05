# name: single_example_multiple_targets_rate
# description: Yes rate on one-example sets (example > 1) for non-example targets that are multiples of the example (10->80, 11->55, 6->18); observed above null means the model still under-generalizes a lone example to its multiples rule (below = over-generalizes).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    d = df[n == 1]
    x = d["number_set"].astype(int)
    t = d["target"].astype(int)
    m = (x > 1) & (t != x) & (t % x == 0)
    if m.sum() == 0:
        return 0.0
    return float(d.loc[m, "rating"].mean())
