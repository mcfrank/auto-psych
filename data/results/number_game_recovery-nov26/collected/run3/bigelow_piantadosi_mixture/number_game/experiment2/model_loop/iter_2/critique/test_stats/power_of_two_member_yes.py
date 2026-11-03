# name: power_of_two_member_yes
# description: In >=2-example sets whose examples are all powers of 2 (2..64) and not all members of a smaller powers-of-m rule (m=4,8), yes rate for non-example power-of-2 targets; observed above null means people favour the powers-of-2 rule more than the family-only prior (equal weight for all powers-of-m rules) allows, below means less.

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
    P2 = {2, 4, 8, 16, 32, 64}
    ok = {}
    for s, e in parsed.items():
        allp = len(e) >= 2 and all(x in P2 for x in e)
        tiny = all(x in (4, 16, 64) for x in e) or all(x in (8, 64) for x in e)
        ok[s] = allp and not tiny
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    sel = df["number_set"].astype(str).map(ok).to_numpy(bool)
    nonex = np.array([tt not in e for tt, e in zip(t, ex)])
    return _rate(r, sel & nonex & np.isin(t, list(P2)))
