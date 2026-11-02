# name: far_target_yes_in_tight_sets
# description: Yes rate for targets >=30 away from every example in >=2-example sets whose examples span <=20; observed above null means people say yes to remote numbers more than the lapse-free model allows (a response floor/lapse is missing), below means less.

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
    m = _item_mask(df, lambda e, x: len(e) >= 2 and max(e) - min(e) <= 20 and min(abs(x - v) for v in e) >= 30)
    return _rate(df, m)
