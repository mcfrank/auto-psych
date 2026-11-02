# name: nested_multiple_specificity
# description: For sets of >=2 examples whose largest common divisor m in 3..12 has a proper divisor d>1 (e.g. all multiples of 9 -> d=3), yes rate on non-example multiples of m minus non-example multiples of d that are not multiples of m; observed > null means people prefer the most specific multiplicative rule more strongly than the model's size principle predicts, observed < null means they generalise to the broader rule more.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    m = np.zeros(len(t))
    for k in range(3, 13):
        ok = np.all((np.mod(np.nan_to_num(E, nan=k), k) == 0), axis=1)
        m = np.where(ok, k, m)
    divs = {4: 2, 6: 3, 8: 4, 9: 3, 10: 5, 12: 6}
    d = np.array([divs.get(int(x), 0) for x in m])
    base = (n >= 2) & (d > 0) & ~in_set
    mm = np.where(m > 0, m, 1)
    dd = np.where(d > 0, d, 1)
    a = base & (np.mod(t, mm) == 0)
    b = base & (np.mod(t, dd) == 0) & (np.mod(t, mm) != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
