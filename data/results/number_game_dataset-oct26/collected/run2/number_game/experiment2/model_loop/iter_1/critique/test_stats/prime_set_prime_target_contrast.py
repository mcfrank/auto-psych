# name: prime_set_prime_target_contrast
# description: In sets of 2+ examples that are all primes, yes-rate for non-example prime targets minus non-example composite targets of the same parity as the examples' majority; observed above the null means people abstract a "prime numbers" rule the model (no primality feature) under-produces, below means the model over-produces it.
def test_statistic(df):
    primes = set([p for p in range(2, 101) if all(p % q for q in range(2, int(p ** 0.5) + 1))])
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    info = {}
    for s in sets:
        ex = [int(v) for v in s.split(",")]
        allp = len(ex) >= 2 and all(v in primes for v in ex)
        par = int(round(np.mean([v % 2 for v in ex])))
        info[s] = (1 if allp else 0, par, ",".join(str(v) for v in ex))
    isp = ns.map({s: info[s][0] for s in sets}).values.astype(bool)
    par = ns.map({s: info[s][1] for s in sets}).values.astype(int)
    t = df["target"].astype(int).values
    tstr = df["target"].astype(int).astype(str)
    padded = ("," + ns + ",")
    ident = np.array([("," + a + ",") in b for a, b in zip(tstr.values, padded.values)])
    tp = np.isin(t, list(primes))
    base = isp & ~ident & (t > 1)
    y = df["rating"].astype(float).values
    a = base & tp
    b = base & ~tp & (t % 2 == par)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
