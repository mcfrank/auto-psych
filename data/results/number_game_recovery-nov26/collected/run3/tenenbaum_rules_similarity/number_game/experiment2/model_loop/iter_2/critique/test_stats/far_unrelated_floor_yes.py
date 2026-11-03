# name: far_unrelated_floor_yes
# description: For sets of >=2 examples, yes rate on targets >15 outside the examples' [min,max] that are in none of the main rules (parity, primes, squares, cubes, powers, multiples 3-12, same last digit) containing all examples; observed > null means people say yes to unrelated numbers more often (a lapse/guessing floor) than the model's predictions allow, observed < null means less.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    lo, hi = np.nanmin(E, axis=1), np.nanmax(E, axis=1)
    N = np.arange(1, 101)
    rules = [N[N % 2 == 1], N[N % 2 == 0], np.array([p for p in range(2, 101) if all(p % a for a in range(2, int(p ** 0.5) + 1))]),
             np.arange(1, 11) ** 2, np.array([1, 8, 27, 64])]
    rules += [N[N % k == 0] for k in range(3, 13)]
    rules += [np.array([b ** k for k in range(1, 8) if b ** k <= 100]) for b in range(2, 11)]
    rules += [N[N % 10 == d] for d in range(10)]
    related = np.zeros(len(t), bool)
    for R in rules:
        related |= np.all(np.isin(E, R) | ~valid, axis=1) & np.isin(t, R)
    m = (n >= 2) & ~related & ((lo - t > 15) | (t - hi > 15))
    return float(r[m].mean()) if m.sum() else 0.0
