# name: rule_inconsistent_yes_rate_rule_sets
# description: For sets of >=2 examples that some arithmetic rule fully contains, yes-rate for targets that belong to none of those containing rules; observed > null means people keep a residual tendency to say yes outside every consistent rule that the model (near-zero) misses; observed < null means the model's interval mass leaks too much.
def _rules():
    N = range(1, 101)
    def isprime(n):
        return n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))
    r = {"odd": [n for n in N if n % 2], "even": list(range(2, 101, 2)),
         "primes": [n for n in N if isprime(n)], "squares": [n * n for n in range(1, 11)],
         "cubes": [n ** 3 for n in range(1, 5)], "pow2_not32": [2, 4, 8, 16, 64],
         "pow2_or37": [2, 4, 8, 16, 32, 37, 64]}
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        r["mult%d" % m] = list(range(m, 101, m))
    for m in range(2, 11):
        r["pow%d" % m] = [m ** k for k in range(1, 8) if m ** k <= 100]
    for dd in range(10):
        r["ends%d" % dd] = [n for n in N if n % 10 == dd]
        r["contains%d" % dd] = [n for n in N if str(dd) in str(n)]
    names = list(r)
    mem = np.zeros((len(names), 101), dtype=bool)
    for i, k in enumerate(names):
        mem[i, r[k]] = True
    return names, mem


def _prep(df):
    names, mem = _rules()
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    sets = [np.array([int(x) for x in s.split(",")]) for s in uniq]
    setmask = np.array([mem[:, s].all(axis=1) for s in sets])  # (nsets, nrules) rules containing all examples
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    return names, mem, codes, sets, setmask, t, y


def _rate(y, m):
    return float(y[m].mean()) if m.sum() > 0 else 0.0


def test_statistic(df):
    names, mem, codes, sets, setmask, t, y = _prep(df)
    s_ok = np.array([len(s) >= 2 for s in sets]) & setmask.any(axis=1)
    in_any = (mem[:, t].T & setmask[codes]).any(axis=1)
    return _rate(y, s_ok[codes] & ~in_any)
