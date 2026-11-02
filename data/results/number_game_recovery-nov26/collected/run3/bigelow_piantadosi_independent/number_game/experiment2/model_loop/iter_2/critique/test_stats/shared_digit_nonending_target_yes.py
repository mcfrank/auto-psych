# name: shared_digit_nonending_target_yes
# description: On sets of 2+ examples that all contain some common digit d, yes rate for non-example targets that contain d but do not end in d, minus the yes rate for non-example targets lacking d; observed above null means people use a contains-digit rule more than the model, below means less.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = (~np.isnan(E)).sum(1)
    isex = (E == t[:, None]).any(1)
    return E, t, r, n, isex

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r, n, isex = _ex(df)
    Ei = np.nan_to_num(E, nan=-1).astype(int)
    ti = t.astype(int)
    lift_m1 = np.zeros(len(t), bool); lift_m0 = np.zeros(len(t), bool)
    for d in range(10):
        def has(x):
            return ((x % 10) == d) | (((x // 10) % 10 == d) & (x >= 10)) | ((x == 100) & (d in (0, 1)))
        allc = (has(Ei) | (Ei < 0)).all(1) & (n >= 2)
        lift_m1 |= allc & has(ti) & (ti % 10 != d)
        lift_m0 |= allc & ~has(ti)
    lift_m0 &= ~lift_m1
    return _rate(r, lift_m1 & ~isex) - _rate(r, lift_m0 & ~isex)
