# name: outlier_set_inhull_nonmember_yes
# description: In >=3-example sets where a narrow rule covers all examples but one, yes rate for non-example targets outside that rule lying within the range of the rule-following examples; observed above null means people fill in the magnitude range of noisy sets more than the model (which explains the outlier as a stray and favours the rule) predicts, below means less.

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

RULES = _rules()  # the incumbent model's 45 rules
NARROW = {k: v for k, v in RULES.items() if k not in ("odd", "even") and not k.startswith("has") and len(v) <= 25}
RULE_COUNT = {x: sum(x in r for r in RULES.values()) for x in range(1, 101)}

def _smallest(e, rules):
    best = None
    for k, r in rules.items():
        if all(x in r for x in e) and (best is None or len(r) < len(rules[best])):
            best = k
    return best

def _outlier_info(e):
    """n>=3 set with no narrow rule covering all examples but one covering all but one: (rule, outlier)."""
    if len(e) < 3 or _smallest(e, NARROW) is not None:
        return None
    best = None
    for i in range(len(e)):
        rest = e[:i] + e[i + 1:]
        k = _smallest(rest, NARROW)
        if k is not None and (best is None or len(NARROW[k]) < len(NARROW[best[0]])):
            best = (k, e[i])
    return best


def test_statistic(df):
    def f(e, x):
        i = _outlier_info(e)
        if i is None or x in e or x in NARROW[i[0]]:
            return False
        rest = [v for v in e if v != i[1]]
        return min(rest) < x < max(rest)
    return _rate(df, _item_mask(df, f))
