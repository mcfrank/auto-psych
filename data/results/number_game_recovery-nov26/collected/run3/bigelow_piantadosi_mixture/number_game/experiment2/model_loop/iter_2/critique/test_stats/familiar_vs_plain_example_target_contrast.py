# name: familiar_vs_plain_example_target_contrast
# description: Among trials whose target equals an example of a >=2-example set, yes rate when that example is a familiar number (1, 100, powers of 2, squares, 27, multiples of 10) minus when it is not; observed above null means people discount familiar-number examples as strays less than the familiar-stray model predicts, below means more.

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
    is_ex = np.array([(tt in e) and len(e) >= 2 for tt, e in zip(t, ex)])
    fam = np.array([tt in FAMILIAR for tt in t])
    return _rate(r, is_ex & fam) - _rate(r, is_ex & ~fam)
