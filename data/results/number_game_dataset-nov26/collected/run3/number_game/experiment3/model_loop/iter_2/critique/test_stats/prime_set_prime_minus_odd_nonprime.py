# name: prime_set_prime_minus_odd_nonprime
# description: On 2+ example sets whose examples are all prime, yes rate for prime non-example targets minus yes rate for odd non-prime targets; observed below null means the model credits the prime rule more sharply than people, who barely separate primes from other odd numbers.
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
    P = np.array([k > 1 and all(k % q for q in range(2, int(k ** 0.5) + 1)) for k in range(101)])
    Ei = np.clip(np.nan_to_num(E).astype(int), 0, 100)
    allp = np.all(np.isnan(E) | P[Ei], axis=1)
    ti = t.astype(int)
    base = (n >= 2) & allp & ~is_ex
    return rate(base & P[ti]) - rate(base & ~P[ti] & (ti % 2 == 1))
