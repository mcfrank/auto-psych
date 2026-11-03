# name: prime_set_prime_targets
# description: In sets of 2+ examples that are all prime, among odd non-example targets more than 3 from every example and sharing no final digit with any example, yes-rate for prime minus non-prime targets; observed above the null means people use the 'primes' rule more than the model, below means the model over-weights it.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    primes = np.array([n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1)) for n in range(101)])
    Ei = np.nan_to_num(E, nan=0).astype(int)
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    all_prime = np.all(primes[Ei] | ~valid, axis=1) & (n >= 2)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    share_fd = np.any(np.mod(E, 10) == (t % 10)[:, None], axis=1)
    m = all_prime & (~is_ex) & (t % 2 == 1) & (dist > 3) & (~share_fd)
    a, b = m & primes[t], m & ~primes[t]
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
