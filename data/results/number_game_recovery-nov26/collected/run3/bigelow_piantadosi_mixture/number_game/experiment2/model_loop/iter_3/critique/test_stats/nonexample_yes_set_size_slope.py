# name: nonexample_yes_set_size_slope
# description: OLS slope of yes (0/1) on number of examples (1-4), over targets not among the examples; observed more negative than null means people tighten generalisation with more examples more strongly than the model's size principle, less negative/positive means the model over-tightens.

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
    n = (df["number_set"].astype(str).str.count(",") + 1).to_numpy().astype(float)
    m = ~_item_mask(df, lambda e, x: x in e)
    r = df["rating"].astype(float).to_numpy()
    x, y = n[m], r[m]
    vx = ((x - x.mean()) ** 2).sum()
    return float(((x - x.mean()) * (y - y.mean())).sum() / vx) if vx > 0 else 0.0
