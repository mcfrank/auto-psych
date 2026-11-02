# name: between_set_nonexample_yes_sd
# description: Standard deviation across sets (with >=15 non-example responses) of the per-set yes rate on non-example targets; observed above null means sets differ in breadth more than the model predicts (it is too uniform), below means the model makes sets too extreme.

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
    m = ~_item_mask(df, lambda e, x: x in e)
    sub = df.loc[m, ["number_set", "rating"]]
    g = sub.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 15]
    return float(g["mean"].std()) if len(g) > 1 else 0.0
