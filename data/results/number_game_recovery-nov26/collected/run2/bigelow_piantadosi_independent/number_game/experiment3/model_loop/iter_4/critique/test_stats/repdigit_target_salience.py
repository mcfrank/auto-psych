# name: repdigit_target_salience
# description: Among non-example targets sharing no digit with any example, yes rate for repdigit targets (11,22,...,99) minus non-repdigit targets, averaged over the in-span and out-of-span strata; observed above null means repdigits are endorsed as a salient category the model lacks.
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    ex = df["number_set"].astype(str)
    exl = ex.str.split(",")
    lo = exl.apply(lambda x: min(int(v) for v in x)).to_numpy()
    hi = exl.apply(lambda x: max(int(v) for v in x)).to_numpy()
    digs = ex.str.replace(",", "", regex=False)
    tstr = pd.Series(t).astype(str)
    share = np.zeros(len(df), dtype=bool)
    for dch in "0123456789":
        share |= (digs.str.contains(dch, regex=False).to_numpy() & tstr.str.contains(dch, regex=False).to_numpy())
    isex = np.array([str(a) in b.split(",") for a, b in zip(t, ex)])
    rep = (t >= 11) & (t % 11 == 0)
    inspan = (t >= lo) & (t <= hi)
    r = df["rating"].to_numpy().astype(float)
    base = (~share) & (~isex)
    diffs = []
    for st in (True, False):
        m = base & (inspan == st)
        a, b = m & rep, m & ~rep
        if a.sum() >= 3 and b.sum() >= 3:
            diffs.append(r[a].mean() - r[b].mean())
    return float(np.mean(diffs)) if diffs else 0.0
