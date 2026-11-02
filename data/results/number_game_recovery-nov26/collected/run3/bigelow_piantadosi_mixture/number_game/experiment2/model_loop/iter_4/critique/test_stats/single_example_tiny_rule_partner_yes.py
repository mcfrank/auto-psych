# name: single_example_tiny_rule_partner_yes
# description: In single-example sets, yes rate for targets sharing with the example a rule of <=7 members (e.g. 25->5, 3->81, 10->100), minus yes rate for other targets; observed below null means people use tiny rules (powers of m, cubes) less than the equal rule prior plus size principle predicts, above means more.

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

def _rules():
    N = range(1, 101)
    R = {"odd": [n for n in N if n % 2], "even": [n for n in N if n % 2 == 0],
         "primes": [n for n in N if n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))],
         "squares": [k * k for k in range(1, 11)], "cubes": [k ** 3 for k in range(1, 5)],
         "pow2not32": [2, 4, 8, 16, 64], "pow2or37": [2, 4, 8, 16, 32, 37, 64]}
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        R["mult%d" % m] = list(range(m, 101, m))
    for m in range(2, 11):
        R["pow%d" % m] = [m ** k for k in range(1, 8) if m ** k <= 100]
    for d in range(10):
        R["ends%d" % d] = [n for n in N if n % 10 == d]
        R["has%d" % d] = [n for n in N if str(d) in str(n)]
    return {k: frozenset(v) for k, v in R.items()}

RULES = _rules()
NARROW = {k: v for k, v in RULES.items() if k not in ("odd", "even") and not k.startswith("has")}

def _smallest_cover(e, maxsize):
    best = None
    for k, r in RULES.items():
        if len(r) <= maxsize and all(x in r for x in e) and (best is None or len(r) < len(best)):
            best = r
    return best

def _covers_all(e):
    return [r for r in RULES.values() if all(x in r for x in e)]

def test_statistic(df):
    TINY = [r for r in RULES.values() if len(r) <= 7]
    rel = lambda e, x: any(e[0] in r and x in r for r in TINY)
    m1 = _item_mask(df, lambda e, x: len(e) == 1 and x != e[0] and rel(e, x))
    m0 = _item_mask(df, lambda e, x: len(e) == 1 and x != e[0] and not rel(e, x))
    return _rate(df, m1) - _rate(df, m0)
