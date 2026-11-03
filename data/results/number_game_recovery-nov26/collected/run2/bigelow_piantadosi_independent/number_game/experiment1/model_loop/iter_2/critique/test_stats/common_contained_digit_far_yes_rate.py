# name: common_contained_digit_far_yes_rate
# description: In sets of >=2 examples that all contain some digit d, yes rate for targets containing d that share no final digit with any example and are >10 from every example; t_observed above null_mean means people use "contains digit d" more than the model's (down-weighted) contains-digit family predicts, below means less.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    v = ~np.isnan(s); n = v.sum(axis=1)
    t = df["target"].astype(int).to_numpy()
    tf = t.astype(float)
    sv = np.where(v, s, 0).astype(int)
    common_hit = np.zeros(len(t), dtype=bool)
    for d in range(10):
        # digit d appears in number x (1..100)
        def has(x):
            tens = x // 10 % 10; ones = x % 10; hund = x // 100
            return (ones == d) | ((tens == d) & (x >= 10)) | ((hund == d) & (x >= 100))
        all_have = (has(sv) | ~v).all(axis=1)
        common_hit |= all_have & has(t)
    dist = np.nanmin(np.abs(s - tf[:, None]), axis=1)
    last_share = (np.where(v, sv % 10, -1) == (t % 10)[:, None]).any(axis=1)
    r = df["rating"].to_numpy(dtype=float)
    m = (n >= 2) & common_hit & ~last_share & (dist > 10)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
