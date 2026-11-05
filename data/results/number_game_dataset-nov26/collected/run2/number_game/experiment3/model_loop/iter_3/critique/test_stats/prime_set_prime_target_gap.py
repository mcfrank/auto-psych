# name: prime_set_prime_target_gap
# description: On multi-example sets whose examples are all prime (e.g. 3,7 / 67,31,17,61), yes rate for prime targets minus yes rate for non-prime targets; observed > null means people infer the prime rule more sharply than the model (observed < null: model over-weights primality).
def test_statistic(df):
    P = {n for n in range(2, 101) if all(n % k for k in range(2, int(n ** 0.5) + 1))}
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    ok = np.zeros(len(u), bool)
    for i, s in enumerate(u):
        ex = [int(x) for x in s.split(",")]
        ok[i] = len(ex) >= 2 and all(e in P for e in ex)
    m = ok[inv]
    t = df["target"].astype(int).values
    hit = np.isin(t, list(P))
    r = df["rating"].astype(float).values
    a, b = m & hit, m & ~hit
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
