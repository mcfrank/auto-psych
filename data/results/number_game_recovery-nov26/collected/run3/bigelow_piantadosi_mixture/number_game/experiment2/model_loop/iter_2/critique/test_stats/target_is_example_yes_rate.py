# name: target_is_example_yes_rate
# description: Yes rate on trials whose target equals one of the set's examples; observed below null means people endorse shown examples less than the model (they write off more examples as strays), above means the model over-attributes examples to strays.

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
    ex, _ = _sets(df)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    m = np.array([tt in e for tt, e in zip(t, ex)])
    return _rate(r, m)
