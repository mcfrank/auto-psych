# name: times_table_rule_sharpness
# description: Among sets of 2+ examples that are all multiples of some m in 3,4,6,7,8,9,11,12 (largest such m; multiples of 5/10 excluded), yes rate on non-example multiples of m minus yes rate on non-multiples; observed below null means people endorse the shared times table less sharply than the model (e.g. rejecting multiples of 8 after 8,64), above null means more sharply.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    ex = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    exi = np.where(valid, ex, 0).astype(int)
    t = df["target"].astype(int).to_numpy()
    mm = np.zeros(len(df), dtype=int)
    for m in [3, 4, 6, 7, 8, 9, 11, 12]:
        ok = (np.where(valid, exi % m == 0, True)).all(axis=1)
        mm = np.where(ok, m, mm)
    five = (np.where(valid, exi % 5 == 0, True)).all(axis=1)
    isex = (ex == t[:, None]).any(axis=1)
    sel = (n >= 2) & (mm > 0) & ~five & ~isex
    r = df["rating"].astype(float).to_numpy()
    mult = sel & (t % np.maximum(mm, 1) == 0)
    non = sel & (t % np.maximum(mm, 1) != 0)
    if mult.sum() == 0 or non.sum() == 0:
        return 0.0
    return float(r[mult].mean() - r[non].mean())
