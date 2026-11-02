# name: prime_vs_odd_composite_in_prime_sets
# description: In >=2-example sets whose examples are all prime, yes rate for prime non-example targets minus yes rate for odd composite targets; observed above null means people prefer the prime concept over plain odd more sharply than the model (its family-level prior underweights primes), below means less.

def _item_mask(df, f):
    """Evaluate f(examples_tuple, target) once per unique (set, target) item; map to rows."""
    s = df["number_set"].astype(str).to_numpy()
    t = df["target"].astype(int).to_numpy()
    keys = pd.Series(s).str.cat(pd.Series(t).astype(str), sep="|").to_numpy()
    uk, inv = np.unique(keys, return_inverse=True)
    vals = np.zeros(len(uk), dtype=bool)
    for i, k in enumerate(uk):
        a, b = k.split("|")
        vals[i] = bool(f(tuple(int(x) for x in a.split(",")), int(b)))
    return vals[inv]

def _rate(df, m):
    r = df["rating"].astype(float).to_numpy()
    return float(r[m].mean()) if m.any() else 0.0

PRIMES = set(n for n in range(2, 101) if all(n % a for a in range(2, int(n ** 0.5) + 1)))

def test_statistic(df):
    base = lambda e, x: len(e) >= 2 and all(v in PRIMES for v in e) and x not in e
    mp = _item_mask(df, lambda e, x: base(e, x) and x in PRIMES)
    mo = _item_mask(df, lambda e, x: base(e, x) and x % 2 == 1 and x > 1 and x not in PRIMES)
    return _rate(df, mp) - _rate(df, mo)
