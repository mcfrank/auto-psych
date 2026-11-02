import os
OUT = "/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/iter_3/critique/test_stats"
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

PRIMES = set(n for n in range(2, 101) if all(n % a for a in range(2, int(n ** 0.5) + 1)))
'''
S = {}
S["prime_vs_odd_composite_in_prime_sets"] = ("In >=2-example sets whose examples are all prime, yes rate for prime non-example targets minus yes rate for odd composite targets; observed above null means people prefer the prime concept over plain odd more sharply than the model (its family-level prior underweights primes), below means less.", '''
    base = lambda e, x: len(e) >= 2 and all(v in PRIMES for v in e) and x not in e
    mp = _item_mask(df, lambda e, x: base(e, x) and x in PRIMES)
    mo = _item_mask(df, lambda e, x: base(e, x) and x % 2 == 1 and x > 1 and x not in PRIMES)
    return _rate(df, mp) - _rate(df, mo)
''')
S["far_target_yes_in_tight_sets"] = ("Yes rate for targets >=30 away from every example in >=2-example sets whose examples span <=20; observed above null means people say yes to remote numbers more than the lapse-free model allows (a response floor/lapse is missing), below means less.", '''
    m = _item_mask(df, lambda e, x: len(e) >= 2 and max(e) - min(e) <= 20 and min(abs(x - v) for v in e) >= 30)
    return _rate(df, m)
''')
S["parity_violating_in_hull_minus_parity_matching_far"] = ("In >=2-example same-parity sets, yes rate for opposite-parity targets strictly inside the examples' range minus yes rate for same-parity targets >10 outside the range; observed above null means people rely on magnitude similarity over the parity rule more than the model predicts, below means they follow parity more.", '''
    same = lambda e: len(e) >= 2 and len(set(v % 2 for v in e)) == 1
    mi = _item_mask(df, lambda e, x: same(e) and min(e) < x < max(e) and x % 2 != e[0] % 2)
    mo = _item_mask(df, lambda e, x: same(e) and (x < min(e) - 10 or x > max(e) + 10) and x % 2 == e[0] % 2)
    return _rate(df, mi) - _rate(df, mo)
''')
S["nonexample_yes_set_size_slope"] = ("OLS slope of yes (0/1) on number of examples (1-4), over targets not among the examples; observed more negative than null means people tighten generalisation with more examples more strongly than the model's size principle, less negative/positive means the model over-tightens.", '''
    n = (df["number_set"].astype(str).str.count(",") + 1).to_numpy().astype(float)
    m = ~_item_mask(df, lambda e, x: x in e)
    r = df["rating"].astype(float).to_numpy()
    x, y = n[m], r[m]
    vx = ((x - x.mean()) ** 2).sum()
    return float(((x - x.mean()) * (y - y.mean())).sum() / vx) if vx > 0 else 0.0
''')
S["outlier_near_vs_far_majority_rule_contrast"] = ("In >=3-example sets where all but one example share a final digit or parity, yes rate for non-example targets obeying the majority rule when the outlier lies within 10 of another example minus when it is >10 from all others; observed above null means a nearby outlier undermines the rule less than the model predicts, below means more.", '''
    def info(e):
        if len(e) < 3:
            return None
        for k in (10, 2):
            ks = [v % k for v in e]
            vals, cnt = np.unique(ks, return_counts=True)
            if cnt.max() == len(e) - 1:
                maj = vals[cnt.argmax()]
                out = [v for v in e if v % k != maj][0]
                near = min(abs(out - v) for v in e if v != out) <= 10
                return k, maj, near
        return None
    def f(e, x, want):
        i = info(e)
        return i is not None and x not in e and x % i[0] == i[1] and i[2] == want
    return _rate(df, _item_mask(df, lambda e, x: f(e, x, True))) - _rate(df, _item_mask(df, lambda e, x: f(e, x, False)))
''')
S["final_digit_sets_tens_digit_match_contrast"] = ("In >=2-example sets whose examples all end in digit d, yes rate for targets whose tens digit is d (not ending in d) minus yes rate for other non-d-ending targets; observed above null means people generalise to 'contains d' more than the model's size principle allows, below means less.", '''
    same = lambda e: len(e) >= 2 and len(set(v % 10 for v in e)) == 1
    tens = lambda e, x: x >= 10 and (x // 10) % 10 == e[0] % 10
    mt = _item_mask(df, lambda e, x: same(e) and x % 10 != e[0] % 10 and tens(e, x))
    mo = _item_mask(df, lambda e, x: same(e) and x % 10 != e[0] % 10 and not tens(e, x))
    return _rate(df, mt) - _rate(df, mo)
''')
S["single_example_multiple_divisor_contrast"] = ("In single-example sets, yes rate for targets that are a multiple or divisor of the example minus yes rate for other same-parity targets; observed above null means people use multiplicative relations to the lone example beyond the model's fixed multiples-of-3..12 list, below means less.", '''
    rel = lambda e, x: x % e[0] == 0 or e[0] % x == 0
    m1 = _item_mask(df, lambda e, x: len(e) == 1 and x != e[0] and rel(e, x))
    m0 = _item_mask(df, lambda e, x: len(e) == 1 and x != e[0] and not rel(e, x) and x % 2 == e[0] % 2)
    return _rate(df, m1) - _rate(df, m0)
''')
S["between_set_nonexample_yes_sd"] = ("Standard deviation across sets (with >=15 non-example responses) of the per-set yes rate on non-example targets; observed above null means sets differ in breadth more than the model predicts (it is too uniform), below means the model makes sets too extreme.", '''
    m = ~_item_mask(df, lambda e, x: x in e)
    sub = df.loc[m, ["number_set", "rating"]]
    g = sub.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 15]
    return float(g["mean"].std()) if len(g) > 1 else 0.0
''')
for name, (desc, body) in S.items():
    with open(os.path.join(OUT, name + ".py"), "w") as fh:
        fh.write(f"# name: {name}\n# description: {desc}\n{HELP}\ndef test_statistic(df):{body}")
