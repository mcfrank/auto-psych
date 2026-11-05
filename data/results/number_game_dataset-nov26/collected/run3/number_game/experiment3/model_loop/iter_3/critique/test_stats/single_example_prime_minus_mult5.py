# name: single_example_prime_minus_mult5
# description: On one-example sets, yes rate for non-example targets when the lone example is prime minus when it is a multiple of 5; observed below null means the model generalizes too much from a lone prime (over-weights the prime rule) and/or too little from a lone round number (under-weights multiples of 5/10).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    Ez = np.where(valid, E, 0.0)
    lo = np.where(valid, E, np.inf).min(axis=1)
    hi = np.where(valid, E, -np.inf).max(axis=1)
    is_ex = (valid & (E == t[:, None])).any(axis=1)
    odd = (np.where(valid, Ez % 2, 1) == 1).all(axis=1)
    even = (np.where(valid, Ez % 2, 0) == 0).all(axis=1)
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    x = Ez[:, 0].astype(int)
    isp = np.ones(len(x), dtype=bool) & (x > 1)
    for a in range(2, 11):
        isp &= ~((x % a == 0) & (x != a))
    base = (n == 1) & ~is_ex
    return rate(base & isp) - rate(base & (x % 5 == 0))
