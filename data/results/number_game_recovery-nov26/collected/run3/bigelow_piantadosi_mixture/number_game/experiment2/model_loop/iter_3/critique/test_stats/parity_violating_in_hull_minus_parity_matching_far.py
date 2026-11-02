# name: parity_violating_in_hull_minus_parity_matching_far
# description: In >=2-example same-parity sets, yes rate for opposite-parity targets strictly inside the examples' range minus yes rate for same-parity targets >10 outside the range; observed above null means people rely on magnitude similarity over the parity rule more than the model predicts, below means they follow parity more.

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
    same = lambda e: len(e) >= 2 and len(set(v % 2 for v in e)) == 1
    mi = _item_mask(df, lambda e, x: same(e) and min(e) < x < max(e) and x % 2 != e[0] % 2)
    mo = _item_mask(df, lambda e, x: same(e) and (x < min(e) - 10 or x > max(e) + 10) and x % 2 == e[0] % 2)
    return _rate(df, mi) - _rate(df, mo)
