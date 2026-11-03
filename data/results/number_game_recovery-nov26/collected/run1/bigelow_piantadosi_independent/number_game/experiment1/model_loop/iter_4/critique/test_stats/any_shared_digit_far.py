# name: any_shared_digit_far
# description: Among non-example targets more than 5 from the nearest example, yes rate for targets sharing any digit (ones or tens) with at least one example minus targets sharing none; observed above null means people use digit-level similarity to individual examples more than the model's whole-rule/interval hypotheses predict, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    lo = np.where(valid, ex, np.inf).min(axis=1)
    hi = np.where(valid, ex, -np.inf).max(axis=1)
    def digs(x):
        x = np.nan_to_num(x, nan=-1)
        return [np.where(x >= 0, np.mod(x, 10), -1), np.where(x >= 10, np.mod(np.floor(x / 10), 10), -2), np.where(x >= 100, 1, -3)]
    E = digs(ex); T = digs(t[:, None])
    share = np.zeros(len(t), dtype=bool)
    for a_ in E:
        for b_ in T:
            share |= ((a_ == b_) & (a_ >= 0) & valid).any(axis=1)
    m = nonmember & (nearest > 5)
    a = r[m & share]; b = r[m & ~share]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
