# name: digit_reversal_target
# description: Yes rate for non-example targets that are the digit reversal of a two-digit example (e.g. 38 for 83) minus yes rate for other non-example targets sharing a digit with an example, averaged over in-span/out-of-span strata; observed above null means digit-anagram similarity drives generalization beyond the model's digit concepts.
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    ex = df["number_set"].astype(str)
    exl = ex.str.split(",")
    lo = exl.apply(lambda x: min(int(v) for v in x)).to_numpy()
    hi = exl.apply(lambda x: max(int(v) for v in x)).to_numpy()
    rev = exl.apply(lambda x: set(int(v[::-1]) for v in x if len(v) == 2 and v[1] != "0" and v[0] != v[1]))
    exset = exl.apply(lambda x: set(int(v) for v in x))
    isrev = np.array([a in s for a, s in zip(t, rev)])
    isex = np.array([a in s for a, s in zip(t, exset)])
    digs = ex.str.replace(",", "", regex=False)
    tstr = pd.Series(t).astype(str)
    share = np.zeros(len(df), dtype=bool)
    for dch in "0123456789":
        share |= (digs.str.contains(dch, regex=False).to_numpy() & tstr.str.contains(dch, regex=False).to_numpy())
    inspan = (t >= lo) & (t <= hi)
    r = df["rating"].to_numpy().astype(float)
    diffs = []
    for st in (True, False):
        m = (~isex) & (inspan == st)
        a, b = m & isrev, m & share & ~isrev
        if a.sum() >= 3 and b.sum() >= 3:
            diffs.append(r[a].mean() - r[b].mean())
    return float(np.mean(diffs)) if diffs else 0.0
