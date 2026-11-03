# name: single_example_multiple_divisor_contrast
# description: In single-example sets, yes rate for targets that are a multiple or divisor of the example minus yes rate for other same-parity targets; observed above null means people use multiplicative relations to the lone example beyond the model's fixed multiples-of-3..12 list, below means less.

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
    rel = lambda e, x: x % e[0] == 0 or e[0] % x == 0
    m1 = _item_mask(df, lambda e, x: len(e) == 1 and x != e[0] and rel(e, x))
    m0 = _item_mask(df, lambda e, x: len(e) == 1 and x != e[0] and not rel(e, x) and x % 2 == e[0] % 2)
    return _rate(df, m1) - _rate(df, m0)
