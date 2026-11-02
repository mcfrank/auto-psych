# name: final_digit_vs_arithmetic_rule_preference
# description: In >=2-example sets covered both by an 'ends in d' rule and by an arithmetic narrow rule (multiples/powers/squares/cubes/primes), yes rate for non-example targets only in the final-digit rule minus targets only in the arithmetic rule(s); observed above null means people favour surface final-digit rules over arithmetic ones more than the model's equal prior plus size principle predicts, below means they favour arithmetic rules more.

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
    def f(e, x, want_end):
        if len(e) < 2 or x in e:
            return False
        cov = [k for k, r in NARROW.items() if all(v in r for v in e)]
        ends = [k for k in cov if k.startswith("ends")]
        ar = [k for k in cov if not k.startswith("ends")]
        if not ends or not ar:
            return False
        in_end = any(x in NARROW[k] for k in ends)
        in_ar = any(x in NARROW[k] for k in ar)
        return (in_end and not in_ar) if want_end else (in_ar and not in_end)
    return _rate(df, _item_mask(df, lambda e, x: f(e, x, True))) - _rate(df, _item_mask(df, lambda e, x: f(e, x, False)))
