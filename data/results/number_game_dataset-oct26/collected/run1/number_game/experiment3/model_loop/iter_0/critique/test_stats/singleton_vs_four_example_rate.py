# name: singleton_vs_four_example_rate
# description: Overall yes-rate for one-example sets minus yes-rate for four-example sets; observed > null means the model under-produces the broader generalisation from a single example relative to larger sets (size principle), observed < null the reverse.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",").to_numpy() + 1
    r = df["rating"].to_numpy().astype(float)
    a = n == 1
    b = n == 4
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
