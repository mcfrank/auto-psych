# name: shared_nonlast_digit_contrast
# description: Among sets of 2+ examples that share a common digit but not a common last digit (e.g. 13,91 share 1), yes rate for non-example targets containing a shared digit minus targets containing none; observed > null means people lean on "contains digit" similarity more than the model, observed < null means the model over-uses contains-digit rules.
def test_statistic(df):
    sets = df["number_set"].unique()
    info = {}
    for s in sets:
        l = [int(x) for x in s.split(",")]
        if len(l) < 2 or len({e % 10 for e in l}) == 1:
            continue
        c = set(str(l[0]))
        for e in l[1:]:
            c &= set(str(e))
        if c:
            info[s] = (c, set(l))
    if not info:
        return 0.0
    sub = df[df["number_set"].isin(list(info))]
    items = sub.groupby(["number_set", "target"])["rating"].mean().reset_index()
    has = np.array([bool(set(str(t)) & info[s][0]) for s, t in zip(items["number_set"], items["target"])])
    isex = np.array([t in info[s][1] for s, t in zip(items["number_set"], items["target"])])
    a = items["rating"][has & ~isex]
    b = items["rating"][~has & ~isex]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
