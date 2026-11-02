# name: incoherent_wide_set_yes_rate
# description: Multi-example sets spanning >=50 that no arithmetic or digit rule (odd/even, multiples 3-12, powers, squares, cubes, primes, ends-in/contains digit) fits: yes-rate on non-example targets; observed above null_mean means people generalise rule-less scattered sets more broadly than the model (whose trust/interval mass is too narrow), below means the model over-generalises them.
def test_statistic(df):
    N = list(range(1, 101))
    def isprime(n):
        return n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))
    rules = [lambda n: n % 2 == 1, lambda n: n % 2 == 0, isprime,
             lambda n: n in {k * k for k in range(1, 11)}, lambda n: n in {1, 8, 27, 64}]
    rules += [(lambda m: (lambda n: n % m == 0))(m) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [(lambda m: (lambda n: n in {m ** k for k in range(1, 8)}))(m) for m in range(2, 11)]
    rules += [(lambda d: (lambda n: n % 10 == d))(d) for d in range(10)]
    rules += [(lambda d: (lambda n: str(d) in str(n)))(d) for d in range(10)]
    sets = df["number_set"].astype(str)
    ok = {}
    for s in sets.unique():
        xs = [int(v) for v in s.split(",")]
        ok[s] = len(xs) >= 2 and max(xs) - min(xs) >= 50 and not any(all(f(x) for x in xs) for f in rules)
    sel = sets.map(ok).values.astype(bool)
    t = df["target"].astype(int).values
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").values
    isex = (ex == t[:, None]).any(axis=1)
    m = sel & ~isex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
