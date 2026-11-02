# name: final_digit_sets_tens_digit_match_contrast
# description: In >=2-example sets whose examples all end in digit d, yes rate for targets whose tens digit is d (not ending in d) minus yes rate for other non-d-ending targets; observed above null means people generalise to 'contains d' more than the model's size principle allows, below means less.

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
    same = lambda e: len(e) >= 2 and len(set(v % 10 for v in e)) == 1
    tens = lambda e, x: x >= 10 and (x // 10) % 10 == e[0] % 10
    mt = _item_mask(df, lambda e, x: same(e) and x % 10 != e[0] % 10 and tens(e, x))
    mo = _item_mask(df, lambda e, x: same(e) and x % 10 != e[0] % 10 and not tens(e, x))
    return _rate(df, mt) - _rate(df, mo)
