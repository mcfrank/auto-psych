# name: single_example_shared_digit
# description: In single-example sets with a two-digit example, among non-example targets more than 5 away, yes-rate for targets sharing any digit with the example minus targets sharing none; observed above the null means a lone example's digits drive generalization more than the model's digit rules predict, below means the model over-applies digit rules to one example.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",").to_numpy()
    x = pd.to_numeric(s.where(single, "0"), errors="coerce").fillna(0).astype(int).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    def digits(v):
        out = np.zeros((len(v), 10), dtype=bool)
        idx = np.arange(len(v))
        out[idx, v % 10] = True
        out[idx[v >= 10], (v[v >= 10] // 10) % 10] = True
        out[idx[v == 100], 1] = True
        return out
    share = np.any(digits(x) & digits(t), axis=1)
    m = single & (x >= 10) & (x <= 99) & (t != x) & (np.abs(t - x) > 5)
    a, b = m & share, m & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
