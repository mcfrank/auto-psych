# name: singleton_minus_multi_yes
# description: Yes rate on 1-example sets minus yes rate on 3-4 example sets; observed above null means people generalize more broadly from one example and tighten with more examples (size principle) more than the averaged-similarity model predicts, below null means the reverse.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    r = df["rating"].astype(float)
    a = r[n == 1]
    b = r[n >= 3]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
