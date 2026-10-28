# name: digit_rule_contrast_k_growth
# description: For sets sharing one final digit, (yes-rate matching-digit minus non-matching-digit non-example targets) at k>=3 examples minus the same contrast at k=2; observed above the null means the "ends in d" rule strengthens with more examples faster than the model's (k-1)*log10 scaling, below means people already apply it fully at k=2 (the model under-applies it for pairs / over-scales with k).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    dig, kk = {}, {}
    for s in sets:
        ex = [int(v) for v in s.split(",")]
        ds = set(v % 10 for v in ex)
        dig[s] = (ex[0] % 10) if (len(ex) >= 2 and len(ds) == 1) else -1
        kk[s] = len(ex)
    d = ns.map(dig).values.astype(int)
    k = ns.map(kk).values.astype(int)
    t = df["target"].astype(int).values
    padded = ("," + ns + ",")
    tstr = df["target"].astype(int).astype(str)
    ident = np.array([("," + a + ",") in b for a, b in zip(tstr.values, padded.values)])
    y = df["rating"].astype(float).values
    def contrast(mask):
        a = mask & (t % 10 == d)
        b = mask & (t % 10 != d)
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return y[a].mean() - y[b].mean()
    base = (d >= 0) & ~ident
    return float(contrast(base & (k >= 3)) - contrast(base & (k == 2)))
