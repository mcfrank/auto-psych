import os
OUT = "/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/test_stats"
os.makedirs(OUT, exist_ok=True)
HELP = '''
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
'''
S = {}
S["familiar_outlier_excuse_contrast"] = ("In >=3-example sets where all but one example share a final digit or a parity, yes rate for non-example targets obeying that majority rule when the lone outlier is a familiar number (1, 100, power of 2, square, 27, multiple of 10) minus when it is unremarkable; observed below null means people do not excuse familiar outliers more than plain ones as much as the familiar-stray model predicts, above means they excuse them even more.", '''
    FAM = set([1, 100] + [2 ** k for k in range(1, 7)] + [k * k for k in range(2, 10)] + [27] + list(range(10, 100, 10)))
    def info(e):
        if len(e) < 3:
            return None
        for k in (10, 2):
            ks = [v % k for v in e]
            vals, cnt = np.unique(ks, return_counts=True)
            if cnt.max() == len(e) - 1:
                maj = vals[cnt.argmax()]
                out = [v for v in e if v % k != maj][0]
                return k, maj, out in FAM
        return None
    def f(e, x, want):
        i = info(e)
        return i is not None and x not in e and x % i[0] == i[1] and i[2] == want
    return _rate(df, _item_mask(df, lambda e, x: f(e, x, True))) - _rate(df, _item_mask(df, lambda e, x: f(e, x, False)))
''')
S["clean_small_rule_member_yes"] = ("In >=3-example sets whose examples all belong to some rule of <=25 members, yes rate for non-example members of the smallest such rule; observed above null means people trust a clean rule more (endorse its members more) than the model's uncertain-reliability/interval mixture predicts, below means less.", '''
    def f(e, x):
        if len(e) < 3 or x in e:
            return False
        r = _smallest_cover(e, 25)
        return r is not None and x in r
    return _rate(df, _item_mask(df, f))
''')
S["clean_rule_inhull_nonmember_yes"] = ("In >=2-example sets whose examples all belong to some rule of <=25 members, yes rate for targets strictly inside the examples' range that belong to no rule containing all examples; observed above null means people let magnitude similarity override a clean rule more than the model's interval component predicts, below means they reject such in-range non-members more strictly.", '''
    def f(e, x):
        if len(e) < 2 or x in e or not (min(e) < x < max(e)):
            return False
        if _smallest_cover(e, 25) is None:
            return False
        return not any(x in r for r in _covers_all(e))
    return _rate(df, _item_mask(df, f))
''')
S["clean_rule_member_inside_minus_outside_hull"] = ("In >=2-example sets whose examples all belong to some rule of <=25 members, yes rate for non-example members of the smallest such rule lying inside the examples' range minus those lying outside it; observed above null means people restrict a rule to the examples' magnitude range more than the model (which adds range only via its separate interval hypotheses), below means less.", '''
    def f(e, x, inside):
        if len(e) < 2 or x in e:
            return False
        r = _smallest_cover(e, 25)
        if r is None or x not in r:
            return False
        return (min(e) < x < max(e)) == inside
    return _rate(df, _item_mask(df, lambda e, x: f(e, x, True))) - _rate(df, _item_mask(df, lambda e, x: f(e, x, False)))
''')
S["unsupported_familiar_target_bias"] = ("Among non-example targets not supported by the examples (every non-parity, non-contains-digit rule containing the target holds fewer than half the examples), yes rate for familiar targets (1, 100, squares, cubes, powers of 2, multiples of 10) minus other targets; observed above null means people endorse familiar numbers regardless of the examples (a target-familiarity bias the model lacks), below means they avoid them.", '''
    FAM = set([1, 100] + [k * k for k in range(1, 11)] + [8, 27, 64] + [2 ** k for k in range(1, 7)] + list(range(10, 101, 10)))
    def unsup(e, x):
        if x in e:
            return False
        for r in NARROW.values():
            if x in r and sum(v in r for v in e) * 2 >= len(e):
                return False
        return True
    mf = _item_mask(df, lambda e, x: unsup(e, x) and x in FAM)
    mo = _item_mask(df, lambda e, x: unsup(e, x) and x not in FAM)
    return _rate(df, mf) - _rate(df, mo)
''')
S["outlier_neighbourhood_generalisation"] = ("In >=3-example sets where a rule of <=25 members (not parity/contains-digit) holds all but one example, yes rate for targets outside that rule within 5 of the outlier minus yes rate for targets outside the rule more than 10 from every example; observed above null means people generalise from the stray example's neighbourhood (treat it as informative) more than the model, which explains it as noise, below means less.", '''
    def info(e):
        if len(e) < 3:
            return None
        best = None
        for r in NARROW.values():
            if len(r) <= 25 and sum(v in r for v in e) == len(e) - 1 and (best is None or len(r) < len(best)):
                best = r
        if best is None:
            return None
        return best, [v for v in e if v not in best][0]
    def f(e, x, near):
        i = info(e)
        if i is None or x in e or x in i[0]:
            return False
        if near:
            return abs(x - i[1]) <= 5
        return min(abs(x - v) for v in e) > 10
    return _rate(df, _item_mask(df, lambda e, x: f(e, x, True))) - _rate(df, _item_mask(df, lambda e, x: f(e, x, False)))
''')
S["full_cover_broad_vs_partial_narrow_contrast"] = ("In >=3-example sets where a smaller rule holds exactly all-but-one example and a larger rule (<=34 members) holds every example, yes rate for non-example targets in the larger rule only minus targets in the smaller rule only; observed above null means people prefer the rule explaining every example over the tighter rule-plus-stray more than the model's reliability belief predicts, below means they accept a stray to keep the tighter rule more readily.", '''
    def info(e):
        if len(e) < 3:
            return None
        full = [r for r in RULES.values() if len(r) <= 34 and all(v in r for v in e)]
        if not full:
            return None
        B = min(full, key=len)
        part = [r for r in RULES.values() if len(r) < len(B) and sum(v in r for v in e) == len(e) - 1]
        if not part:
            return None
        return B, min(part, key=len)
    def f(e, x, broad):
        i = info(e)
        if i is None or x in e:
            return False
        B, Nr = i
        return (x in B and x not in Nr) if broad else (x in Nr and x not in B)
    return _rate(df, _item_mask(df, lambda e, x: f(e, x, True))) - _rate(df, _item_mask(df, lambda e, x: f(e, x, False)))
''')
S["single_example_tiny_rule_partner_yes"] = ("In single-example sets, yes rate for targets sharing with the example a rule of <=7 members (e.g. 25->5, 3->81, 10->100), minus yes rate for other targets; observed below null means people use tiny rules (powers of m, cubes) less than the equal rule prior plus size principle predicts, above means more.", '''
    TINY = [r for r in RULES.values() if len(r) <= 7]
    rel = lambda e, x: any(e[0] in r and x in r for r in TINY)
    m1 = _item_mask(df, lambda e, x: len(e) == 1 and x != e[0] and rel(e, x))
    m0 = _item_mask(df, lambda e, x: len(e) == 1 and x != e[0] and not rel(e, x))
    return _rate(df, m1) - _rate(df, m0)
''')
for name, (desc, body) in S.items():
    with open(os.path.join(OUT, name + ".py"), "w") as fh:
        fh.write("# name: %s\n# description: %s\n%s\ndef test_statistic(df):%s" % (name, desc, HELP, body))
