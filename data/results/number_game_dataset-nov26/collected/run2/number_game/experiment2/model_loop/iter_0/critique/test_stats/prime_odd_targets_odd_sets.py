# name: prime_odd_targets_odd_sets
# description: In sets whose examples are all odd but not all prime, among odd targets, yes-rate for prime targets minus non-prime odd targets; observed > null means people favour prime-looking targets beyond what the model (prime family only active when all examples are prime) produces.
def test_statistic(df):
    primes = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}
    ok = {}
    for s in df["number_set"].unique():
        ex = [int(x) for x in str(s).split(",")]
        ok[s] = all(e % 2 == 1 for e in ex) and not all(e in primes for e in ex)
    t = df["target"].astype(int)
    sel = df["number_set"].map(ok).astype(bool).values & (t.values % 2 == 1)
    if sel.sum() == 0:
        return 0.0
    tp = t.values[sel]
    isp = np.isin(tp, list(primes))
    r = df["rating"].values[sel].astype(float)
    if isp.sum() == 0 or (~isp).sum() == 0:
        return 0.0
    return float(r[isp].mean() - r[~isp].mean())
