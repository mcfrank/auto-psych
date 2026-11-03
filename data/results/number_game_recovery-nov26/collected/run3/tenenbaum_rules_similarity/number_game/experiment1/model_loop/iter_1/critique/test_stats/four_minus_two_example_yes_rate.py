# name: four_minus_two_example_yes_rate
# description: Yes-rate on 4-example sets minus yes-rate on 2-example sets; observed > model means people narrow their generalisation with more examples less than the size principle predicts (observed < model: they narrow more).
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",").to_numpy() + 1
    r = df["rating"].to_numpy().astype(float)
    a, b = n == 4, n == 2
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
