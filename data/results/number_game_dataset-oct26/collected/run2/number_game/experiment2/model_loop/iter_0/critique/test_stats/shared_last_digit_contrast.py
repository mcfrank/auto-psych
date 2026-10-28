# name: shared_last_digit_contrast
# description: In sets of 2+ examples that all end in the same digit, yes-rate for (non-identical) targets ending in that digit minus targets ending in another digit; observed above the null means people apply an explicit "ends in d" rule more sharply than the model's final-digit similarity dimension predicts, below means less.
def test_statistic(df):
    sets = df["number_set"].astype(str).unique()
    dig = {}
    for s in sets:
        ex = [int(v) for v in s.split(",")]
        ds = set(v % 10 for v in ex)
        dig[s] = (ds.pop() if (len(ex) >= 2 and len(ds) == 1) else -1)
    ns = df["number_set"].astype(str)
    d = ns.map(dig).values.astype(int)
    exs = ns.map({s: set(int(v) for v in s.split(",")) for s in sets})
    t = df["target"].astype(int).values
    ident = np.array([tt in e for tt, e in zip(t, exs)])
    base = (d >= 0) & ~ident
    match = base & (t % 10 == d)
    other = base & (t % 10 != d)
    r = df["rating"].astype(float).values
    if match.sum() == 0 or other.sum() == 0:
        return 0.0
    return float(r[match].mean() - r[other].mean())
