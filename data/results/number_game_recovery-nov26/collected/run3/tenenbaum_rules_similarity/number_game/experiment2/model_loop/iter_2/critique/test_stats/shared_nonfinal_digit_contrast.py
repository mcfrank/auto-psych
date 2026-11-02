# name: shared_nonfinal_digit_contrast
# description: For sets of >=2 examples that all contain some digit d but do not all end in the same digit (e.g. 83,38,58,68), among non-example targets outside the examples' [min,max], yes rate on targets containing d minus targets not containing d; observed > null means people use a "contains digit d" concept more than the model's family prior predicts, observed < null less.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    lo, hi = np.nanmin(E, axis=1), np.nanmax(E, axis=1)
    Ez = np.nan_to_num(E, nan=-1)
    last = np.where(valid, np.mod(Ez, 10), np.nan)
    same_last = np.nanmin(last, axis=1) == np.nanmax(last, axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    base = (n >= 2) & ~same_last & ~in_set & ((t < lo) | (t > hi))
    a = np.zeros(len(t), bool)
    b = np.zeros(len(t), bool)
    for d in range(10):
        has = (np.mod(Ez, 10) == d) | ((np.floor(Ez / 10) % 10 == d) & (Ez >= 10))
        alld = np.all(has | ~valid, axis=1)
        th = (np.mod(t, 10) == d) | ((np.floor(t / 10) % 10 == d) & (t >= 10))
        a |= base & alld & th
        b |= base & alld & ~th
    b &= ~a
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
