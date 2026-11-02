# name: prime_set_prime_vs_odd_composite
# description: Among sets of 2+ examples that are all odd primes, yes rate for prime targets minus odd composite targets (non-examples); observed > null means people pick out "primes" over "odd numbers" more sharply than the model, observed < null means the model over-commits to the primes rule.
def test_statistic(df):
    def isp(n):
        return n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))
    primes = {n for n in range(1, 101) if isp(n)}
    sets = df["number_set"].unique()
    keep = {}
    for s in sets:
        l = [int(x) for x in s.split(",")]
        if len(l) >= 2 and all(e in primes and e > 2 for e in l):
            keep[s] = set(l)
    if not keep:
        return 0.0
    sub = df[df["number_set"].isin(list(keep))]
    items = sub.groupby(["number_set", "target"])["rating"].mean().reset_index()
    t = items["target"].astype(int).to_numpy()
    isex = np.array([tt in keep[s] for s, tt in zip(items["number_set"], t)])
    ip = np.isin(t, list(primes))
    oc = (t % 2 == 1) & ~ip & (t > 1)
    a = items["rating"][ip & ~isex]
    b = items["rating"][oc & ~isex]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
