# name: any_shared_digit_far
# description: Among non-example targets more than 5 from the nearest example, yes rate for targets sharing any digit (either position) with at least one example minus targets sharing no digit with any example; observed above null means people use digit-level similarity to individual examples more than the model's whole-rule hypotheses predict.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    def digits(x):
        x = np.nan_to_num(x, nan=-1)
        o = np.where(x >= 0, np.mod(x, 10), -1)
        tdig = np.where(x >= 10, np.mod(np.floor(x / 10), 10), -2)
        h = np.where(x >= 100, 1, -3)
        return o, tdig, h
    eo, et, eh = digits(ex)
    to, tt, th = digits(t[:, None])
    share = np.zeros(len(t), dtype=bool)
    for a in (eo, et, eh):
        for b in (to, tt, th):
            share |= ((a == b) & (a >= 0) & valid).any(axis=1)
    m = nonmember & (nearest > 5)
    a_, b_ = r[m & share], r[m & ~share]
    if len(a_) == 0 or len(b_) == 0:
        return 0.0
    return float(a_.mean() - b_.mean())
