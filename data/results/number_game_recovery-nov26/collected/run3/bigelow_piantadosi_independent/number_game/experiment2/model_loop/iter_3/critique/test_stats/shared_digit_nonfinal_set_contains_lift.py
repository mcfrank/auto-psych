# name: shared_digit_nonfinal_set_contains_lift
# description: Among sets of 3+ examples that all contain a common digit d but do not all end in d (e.g. 83,38,58,68), yes rate for non-example targets containing d minus targets not containing d; observed above null means people generalise "contains d" more than the model, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = (~np.isnan(ex)).sum(1)
    is_ex = (ex == t[:, None]).any(1)
    P = set(n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1)))
    def m(mask):
        return float(y[mask].mean()) if mask.sum() > 0 else 0.0
    e = np.nan_to_num(ex, nan=-1).astype(int)
    tt = t.astype(int)
    lift_num = np.zeros(len(t), dtype=bool); sel_any = np.zeros(len(t), dtype=bool)
    out = []
    for d in range(10):
        has = ((e % 10 == d) | ((e // 10) % 10 == d) & (e >= 10) | (e == 100) & (d in (0, 1)))
        has = has | (e < 0)
        ends = (e % 10 == d) | (e < 0)
        s = (k >= 3) & has.all(1) & ~ends.all(1) & ~is_ex
        tc = (tt % 10 == d) | (((tt // 10) % 10 == d) & (tt >= 10)) | ((tt == 100) & (d in (0, 1)))
        sel_any |= s
        lift_num |= s & tc
    return m(lift_num) - m(sel_any & ~lift_num)
