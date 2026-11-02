# name: set_size_yes_rate_difference
# description: Yes-rate on 4-example sets minus yes-rate on 1-2-example sets; observed less negative (closer to 0) than null means the model's size principle over-produces narrowing of generalization with more examples, more negative means it under-produces it.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",").values + 1
    r = df["rating"].astype(float).values
    a, b = r[n == 4], r[n <= 2]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
