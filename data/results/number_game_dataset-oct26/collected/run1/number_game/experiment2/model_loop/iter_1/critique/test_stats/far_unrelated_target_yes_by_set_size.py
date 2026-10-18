# name: far_unrelated_target_yes_by_set_size
# description: For "unrelated" targets (no shared final digit, no shared decade, not a multiple of the set gcd>=2), yes rate in 1-example sets minus yes rate in 3-4-example sets; t_observed > null_mean means people generalize more loosely from one example and tighten with more examples (size principle) more than the model predicts, < means the reverse.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    codes, uniq = pd.factorize(sets)
    X = np.full((len(uniq), 4), -1000)
    G = np.ones(len(uniq), dtype=int)
    N = np.zeros(len(uniq), dtype=int)
    for i, s in enumerate(uniq):
        xs = [int(v) for v in s.split(",")][:4]
        X[i, :len(xs)] = xs
        g = 0
        for v in xs:
            g = math.gcd(g, v)
        G[i] = max(g, 1)
        N[i] = len(xs)
    E = X[codes]
    valid = E > -1000
    same_dec = (valid & ((E // 10) == (t // 10)[:, None])).any(1)
    same_dig = (valid & ((E % 10) == (t % 10)[:, None])).any(1)
    g = G[codes]
    mult = (g >= 2) & (t % g == 0)
    unrel = ~same_dec & ~same_dig & ~mult
    n = N[codes]
    r = df["rating"].values.astype(float)
    a = unrel & (n == 1)
    b = unrel & (n >= 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
