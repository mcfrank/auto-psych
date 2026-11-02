# name: single_example_multiple_yes
# description: For single-example sets, yes rate for non-example targets that are multiples of the example's smallest factor >1 minus yes rate for non-multiples; observed above null means one example triggers rule generalisation more than the model predicts.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = ~s.str.contains(",").values
    x = pd.to_numeric(s.where(one, "0")).astype(int).values
    t = df["target"].astype(int).values
    f = np.zeros_like(x)
    for k in range(9, 1, -1):
        f = np.where((x > 1) & (x % k == 0), k, f)
    ok = one & (f > 1) & (t != x)
    mul = ok & (t % np.maximum(f, 1) == 0)
    r = df["rating"].values
    return float(r[mul].mean() - r[ok & ~mul].mean())
