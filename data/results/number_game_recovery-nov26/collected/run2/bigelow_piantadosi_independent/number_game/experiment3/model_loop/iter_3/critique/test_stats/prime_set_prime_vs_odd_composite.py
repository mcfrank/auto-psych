# name: prime_set_prime_vs_odd_composite
# description: Among sets whose 2+ examples are all primes, yes-rate on prime non-example targets minus yes-rate on odd composite targets; observed above the model means people discriminate primes from merely odd numbers more sharply than the model's (search-limited) prime rule predicts, below means less.
def _set_info(df):
    # Per distinct set: number of examples, span, examples, and the computed rules all examples fit.
    primes = [n for n in range(2, 101) if all(n % a for a in range(2, int(n ** 0.5) + 1))]
    rules = [primes, [2, 4, 8, 16, 32, 64], [i * i for i in range(1, 11)], [1, 8, 27, 64]]
    rules += [list(range(m, 101, m)) for m in [3, 4, 6, 7, 8, 9, 11, 12]]
    rules = [set(r) for r in rules]
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    k = len(uniq)
    n = np.zeros(k); lo = np.zeros(k); hi = np.zeros(k)
    member = np.zeros((k, 102), bool); example = np.zeros((k, 102), bool)
    for i, s in enumerate(uniq):
        e = [int(x) for x in s.split(",")]
        n[i], lo[i], hi[i] = len(e), min(e), max(e)
        example[i, e] = True
        for r in rules:
            if all(x in r for x in e):
                member[i, list(r)] = True
    t = df["target"].astype(int).to_numpy()
    computed = member.any(axis=1)[codes]
    return dict(codes=codes, n=n[codes], lo=lo[codes], hi=hi[codes], t=t,
                member=member[codes, t], example=example[codes, t], computed=computed,
                y=df["rating"].astype(float).to_numpy())

def test_statistic(df):
    d = _set_info(df)
    t = d["t"]
    isprime = np.array([False, False] + [all(k % a for a in range(2, int(k ** 0.5) + 1)) for k in range(2, 102)])[t]
    allprime = np.zeros(len(t), bool)
    s = df["number_set"].astype(str)
    allp = s.map(lambda z: all(int(x) > 1 and all(int(x) % a for a in range(2, int(int(x) ** 0.5) + 1)) for x in z.split(",")))
    allprime = allp.to_numpy() & (d["n"] >= 2)
    a = allprime & isprime & ~d["example"]
    b = allprime & ~isprime & (t % 2 == 1) & (t > 1)
    if not a.any() or not b.any():
        return 0.0
    return float(d["y"][a].mean() - d["y"][b].mean())
