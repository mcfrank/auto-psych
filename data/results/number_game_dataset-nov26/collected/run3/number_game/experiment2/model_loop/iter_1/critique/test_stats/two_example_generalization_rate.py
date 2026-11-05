# name: two_example_generalization_rate
# description: Yes rate for non-example targets on sets with exactly two examples; observed above null means the model's size principle/range prior is too strict with two examples (below = too lenient).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    d = df[n == 2]
    ex = d["number_set"].str.split(",")
    t = d["target"].astype(int)
    a = ex.str[0].astype(int); b = ex.str[1].astype(int)
    m = (t != a) & (t != b)
    return float(d.loc[m, "rating"].mean()) if m.sum() else 0.0
