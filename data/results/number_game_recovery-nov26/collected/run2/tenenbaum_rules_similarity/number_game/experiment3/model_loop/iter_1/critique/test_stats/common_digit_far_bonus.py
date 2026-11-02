# name: common_digit_far_bonus
# description: Among multi-example sets, for targets more than 10 from every example and not ending in any example's final digit, yes-rate when the target contains a digit written in every example minus yes-rate otherwise; observed above null means the model under-produces digit-containment generalisation.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    ok = ~np.isnan(ex)
    def mask(v):
        v = v.astype(np.int64)
        tensd = (v // 10) % 10
        m = np.left_shift(1, v % 10)
        m = m | np.where(v >= 10, np.left_shift(1, tensd), 0)
        m = m | np.where(v >= 100, 2, 0)
        return m
    exm = np.where(ok, mask(np.nan_to_num(ex, nan=1.0)), 1023)
    common = np.bitwise_and.reduce(exm, axis=1)
    tm = mask(t)
    has = (common & tm) != 0
    n = ok.sum(axis=1)
    dist = np.where(ok, np.abs(ex - t[:, None]), np.inf).min(axis=1)
    last = np.where(ok, (np.nan_to_num(ex) % 10) == (t % 10)[:, None], False).any(axis=1)
    m = (n >= 2) & (dist > 10) & ~last
    a, b = m & has, m & ~has
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
