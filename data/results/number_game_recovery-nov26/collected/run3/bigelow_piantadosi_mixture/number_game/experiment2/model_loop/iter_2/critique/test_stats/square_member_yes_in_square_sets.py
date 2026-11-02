# name: square_member_yes_in_square_sets
# description: In >=2-example sets whose examples are all perfect squares (1..100) but not all in one powers-of-m rule, yes rate for non-example square targets; observed above null means people favour the squares rule more than the model's family-level prior predicts, below means less.

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
    SQ = set(k * k for k in range(1, 11))
    tiny = [set(m ** k for k in range(1, 8) if m ** k <= 100) for m in range(2, 11)]
    ok = {s: len(e) >= 2 and all(x in SQ for x in e) and not any(all(x in T for x in e) for T in tiny) for s, e in parsed.items()}
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    sel = df["number_set"].astype(str).map(ok).to_numpy(bool)
    nonex = np.array([tt not in e for tt, e in zip(t, ex)])
    return _rate(r, sel & nonex & np.isin(t, list(SQ)))
