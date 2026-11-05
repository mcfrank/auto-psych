# name: ruleless_three_plus_in_range_rate
# description: On 3-4 example sets that no rule (odd/even, multiples 3-12, powers 2-10, squares, cubes, primes, ends-in/contains digit) fits fully, yes rate for in-range non-example targets; observed above null means the model under-generalizes within the examples' range when no rule explains the examples.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    U = E % 10; tu = t % 10
    def rate(m):
        return float(y[m].mean()) if m.any() else 0.0
    def fits(member):
        ok = np.ones(len(t), dtype=bool)
        for j in range(4):
            v = E[:, j]
            ok &= np.isnan(v) | member(np.nan_to_num(v).astype(int))
        return ok
    any_fit = fits(lambda v: v % 2 == 1) | fits(lambda v: v % 2 == 0)
    for mm in range(3, 13):
        any_fit |= fits(lambda v, mm=mm: v % mm == 0)
    sets = [{i * i for i in range(1, 11)}, {1, 8, 27, 64},
            {p for p in range(2, 101) if all(p % q for q in range(2, int(p ** 0.5) + 1))}]
    for b in range(2, 11):
        sets.append({b ** k for k in range(1, 8) if b ** k <= 100})
    for s in sets:
        arr = np.zeros(101, dtype=bool); arr[list(s)] = True
        any_fit |= fits(lambda v, arr=arr: arr[np.clip(v, 0, 100)])
    for d in range(10):
        any_fit |= fits(lambda v, d=d: v % 10 == d)
        any_fit |= fits(lambda v, d=d: (v % 10 == d) | ((v // 10) % 10 == d) | ((d == 0) & (v == 100)) | ((d == 1) & (v == 100)))
    m = (n >= 3) & ~any_fit & ~is_ex & (t >= lo) & (t <= hi)
    return rate(m)
