# name: familiar_vs_plain_outlier_rule_contrast
# description: In >=3-example sets where all but one example share a final digit or a parity, yes rate for non-example targets obeying that majority rule when the lone outlier is a familiar number (1, 100, power of 2, square, 27, multiple of 10) minus when it is unremarkable; observed below null means people do not excuse familiar outliers more than plain ones as strongly as the familiar-stray model predicts, above means they excuse them even more.

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
    info = {}
    for s, e in parsed.items():
        info[s] = None
        n = len(e)
        if n < 3:
            continue
        for key in (lambda x: x % 10, lambda x: x % 2):
            ks = [key(x) for x in e]
            vals, cnt = np.unique(ks, return_counts=True)
            if cnt.max() == n - 1:
                maj = vals[cnt.argmax()]
                out = [x for x in e if key(x) != maj][0]
                info[s] = (key, maj, out in FAMILIAR)
                break
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    sets = df["number_set"].astype(str).to_numpy()
    obey = np.zeros(len(t), bool); fam = np.zeros(len(t), bool)
    for s in parsed:
        if info[s] is None:
            continue
        key, maj, f = info[s]
        m = (sets == s) & ~np.isin(t, parsed[s])
        ok = (t % 10 == maj) if key(13) == 3 else (t % 2 == maj)
        obey |= m & ok
        fam |= m & ok & f
    return _rate(r, obey & fam) - _rate(r, obey & ~fam)
