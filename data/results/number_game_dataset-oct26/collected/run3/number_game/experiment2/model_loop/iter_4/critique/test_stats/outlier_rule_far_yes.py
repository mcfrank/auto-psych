# name: outlier_rule_far_yes
# description: For 3-4 example sets where all but exactly one example share a last digit, yes-rate for out-of-range targets with that shared last digit; observed above null means the model is too intolerant of the exception (under-keeps the near-fitting rule), below means too tolerant.
def test_statistic(df):
    s = df["number_set"].astype(str)
    sp = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    v = sp.values.astype(float)
    fin = np.isfinite(v)
    n = fin.sum(axis=1)
    lo = np.nanmin(v, axis=1); hi = np.nanmax(v, axis=1)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    ld = np.where(fin, np.mod(np.nan_to_num(v), 10), -1)
    best_cnt = np.zeros(len(t)); best_d = np.full(len(t), -1.0)
    for d in range(10):
        c = (ld == d).sum(axis=1)
        upd = c > best_cnt
        best_cnt = np.where(upd, c, best_cnt); best_d = np.where(upd, d, best_d)
    m = (n >= 3) & (best_cnt == n - 1) & (np.mod(t, 10) == best_d) & ((t < lo) | (t > hi))
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
