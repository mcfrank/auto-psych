# name: single_example_multiples_rate
# description: Yes rate on one-example sets for targets that are a multiple of the lone example (target != example); observed above null means the model under-generalizes a lone example to its multiples (rule belief too diffuse).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    s = df[n == 1]
    x = s["number_set"].astype(int)
    t = s["target"].astype(int)
    m = (t % x == 0) & (t != x)
    if m.sum() == 0:
        return 0.0
    return float(s.loc[m, "rating"].mean())
