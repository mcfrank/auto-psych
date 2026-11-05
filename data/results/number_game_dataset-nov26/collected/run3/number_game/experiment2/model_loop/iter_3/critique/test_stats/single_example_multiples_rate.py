# name: single_example_multiples_rate
# description: Yes rate on one-example sets (example > 1) for non-example targets that are multiples of the example (6->18, 11->55, 25->100); observed above null means the model, even with digit rules discounted, still under-generalizes a lone example to its multiples (below = over-generalizes).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    sub = df[n == 1]
    x = sub["number_set"].astype(int)
    t = sub["target"].astype(int)
    m = (x > 1) & (t != x) & (t % x == 0)
    if m.sum() == 0:
        return 0.0
    return float(sub["rating"][m].mean())
