# name: gcd_multiple_contrast
# description: In >=2-example sets whose examples' gcd is >=3, yes rate for non-example targets that are multiples of the gcd minus yes rate for non-multiples (both excluding targets sharing a final digit with any example); observed above null means people generalise the multiples rule more sharply than the model predicts, below means less sharply.

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
    g = {s: (math.gcd(*e) if len(e) >= 2 else 0) for s, e in parsed.items()}
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    gg = df["number_set"].astype(str).map(g).to_numpy(int)
    clean = np.array([(tt not in e) and all(tt % 10 != x % 10 for x in e) for tt, e in zip(t, ex)])
    sel = (gg >= 3) & clean
    mult = t % np.maximum(gg, 1) == 0
    return _rate(r, sel & mult) - _rate(r, sel & ~mult)
