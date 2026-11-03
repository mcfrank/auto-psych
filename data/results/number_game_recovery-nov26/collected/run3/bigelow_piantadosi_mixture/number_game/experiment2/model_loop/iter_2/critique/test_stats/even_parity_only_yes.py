# name: even_parity_only_yes
# description: In >=2-example sets whose examples are all even and do not all share a final digit, yes rate for non-example even targets that are NOT multiples of the examples' gcd (when gcd>2) -- i.e. supported only by the 'even' rule or ranges; observed above null means people use the parity rule more than the model predicts, below means less.

def _sets(df):
    """Per row: tuple of examples (shown order). Parsed once per unique set."""
    u = pd.unique(df["number_set"].astype(str))
    parsed = {s: tuple(int(x) for x in s.split(",")) for s in u}
    return df["number_set"].astype(str).map(parsed), parsed

def _rate(r, m):
    m = np.asarray(m, dtype=bool)
    return float(r[m].mean()) if m.any() else 0.0

FAMILIAR = set([1, 100] + [2 ** k for k in range(1, 7)] + [k * k for k in range(2, 10)] + [27] + list(range(10, 100, 10)))

def test_statistic(df):
    ex, parsed = _sets(df)
    g = {}
    for s, e in parsed.items():
        ok = len(e) >= 2 and all(x % 2 == 0 for x in e) and len(set(x % 10 for x in e)) > 1
        g[s] = math.gcd(*e) if ok else 0
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    gg = df["number_set"].astype(str).map(g).to_numpy(int)
    nonex = np.array([tt not in e for tt, e in zip(t, ex)])
    cover = (gg > 2) & (t % np.maximum(gg, 1) == 0)
    return _rate(r, (gg > 0) & nonex & (t % 2 == 0) & ~cover)
