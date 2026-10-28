# name: common_contained_digit_far_effect
# description: For sets of 2+ examples in which some digit appears in every example (any position), yes rate on non-example targets more than 10 from every example that also contain that shared digit; observed above null_mean means people generalize by contained-digit rules more than the model predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n_ex = valid.sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    e = np.where(valid, ex, -1).astype(np.int64)
    tens = np.where(e >= 10, (e // 10) % 10, -1); ones = np.where(e >= 0, e % 10, -1); hund = np.where(e >= 100, 1, -1)
    ti = t.astype(np.int64)
    t_tens = np.where(ti >= 10, (ti // 10) % 10, -1); t_ones = ti % 10
    shared_any = np.zeros(len(t), dtype=bool); t_has = np.zeros(len(t), dtype=bool)
    for d in range(1, 10):
        has = (tens == d) | (ones == d) | (hund == d)
        in_all = np.all(np.where(valid, has, True), axis=1) & (n_ex >= 2)
        shared_any |= in_all
        t_has |= in_all & ((t_tens == d) | (t_ones == d) | ((ti >= 100) & (d == 1)))
    base = shared_any & (~member) & (dist > 10)
    a = base & t_has
    if a.sum() == 0:
        return 0.0
    return float(y[a].mean())
