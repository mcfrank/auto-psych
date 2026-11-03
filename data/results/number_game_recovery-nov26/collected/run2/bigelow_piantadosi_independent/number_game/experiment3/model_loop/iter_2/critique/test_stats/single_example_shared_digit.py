# name: single_example_shared_digit
# description: For 1-example sets, among non-example targets more than 10 away from the example, yes-rate for targets sharing any digit with the example minus targets sharing none; observed above null_mean means people generalize from one number by surface digit overlap more than the model's contains-digit/final-digit rules produce (below = less).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    e = ex[:, 0]
    t = df["target"].astype(int).to_numpy()
    sel = (n == 1) & (np.abs(t - e) > 10)
    e_i = np.where(sel, e, 0).astype(int)
    def digits(v):
        D = np.zeros((len(v), 10), dtype=bool)
        r = np.arange(len(v))
        D[r, v % 10] = True
        tens = (v // 10) % 10
        D[r, tens] |= (v >= 10)
        D[r, 1] |= (v == 100)
        D[r, 0] |= (v == 100)
        return D
    share = (digits(e_i) & digits(t)).any(axis=1)
    y = df["rating"].astype(float).to_numpy()
    a = sel & share
    b = sel & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
