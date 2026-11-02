# name: tiny_power_rule_broad_member_yes
# description: In >=2-example sets whose examples all lie in a tiny powers-of-m rule (m>=3, at most 4 members, e.g. {8,64}, {9,81}, {4,16,64}), yes rate for non-example targets that are powers of 2, powers of 3, squares or cubes but outside that tiny rule; observed above null means people generalise to the broader familiar rule more than the size principle with equal within-family priors predicts, below means less.

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
    tiny = [set(m ** k for k in range(1, 8) if m ** k <= 100) for m in range(3, 11)]
    broad = set([2 ** k for k in range(1, 7)] + [3 ** k for k in range(1, 5)] + [k * k for k in range(1, 11)] + [k ** 3 for k in range(1, 5)])
    rule = {}
    for s, e in parsed.items():
        rule[s] = None
        if len(e) >= 2:
            for T in tiny:
                if all(x in T for x in e):
                    rule[s] = T
                    break
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    rr = df["number_set"].astype(str).map(rule).to_numpy()
    m = np.array([(T is not None) and (tt in broad) and (tt not in T) and (tt not in e) for tt, T, e in zip(t, rr, ex)])
    return _rate(r, m)
