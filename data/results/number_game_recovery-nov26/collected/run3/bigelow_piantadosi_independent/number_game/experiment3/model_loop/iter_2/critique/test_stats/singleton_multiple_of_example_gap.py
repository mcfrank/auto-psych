# name: singleton_multiple_of_example_gap
# description: Among single-example sets with example 2-50, yes rate for targets that are other multiples of the example minus yes rate for non-multiples; observed above null means after one example people already generalise to its times table more than the model predicts (model's weak single-example commitment is too weak), below null means less.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = (~s.str.contains(",")).to_numpy()
    x = pd.to_numeric(s, errors="coerce").fillna(0).to_numpy().astype(int)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    ok = single & (x >= 2) & (x <= 50) & (t != x)
    xs = np.where(x > 0, x, 1)
    mult = (t % xs == 0)
    a = ok & mult
    b = ok & ~mult
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
