# name: incoherent_set_yes_rate
# description: Overall yes rate on multi-example sets with no evident rule (gcd 1, mixed parity, more than one final digit, span > 40); observed < model means people reject most targets when examples share nothing (no rule found), whereas averaged exemplar similarity still yields moderate familiarity; observed > model means the reverse.
def test_statistic(df):
    s = df["number_set"].astype(str)
    fmap = {}
    for u in s.unique():
        v = [int(z) for z in u.split(",")]
        g = 0
        for z in v:
            g = math.gcd(g, z)
        ok = (len(v) >= 2 and g == 1 and len(set(z % 2 for z in v)) > 1
              and len(set(z % 10 for z in v)) > 1 and max(v) - min(v) > 40)
        fmap[u] = ok
    m = s.map(fmap).values.astype(bool)
    y = df["rating"].astype(float).values
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
