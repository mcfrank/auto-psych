# name: outlier_near_vs_far_majority_rule_contrast
# description: In >=3-example sets where all but one example share a final digit or parity, yes rate for non-example targets obeying the majority rule when the outlier lies within 10 of another example minus when it is >10 from all others; observed above null means a nearby outlier undermines the rule less than the model predicts, below means more.

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
