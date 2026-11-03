# name: shared_contained_digit_contrast
# description: Among multi-example sets whose examples all contain some common digit d but do NOT all end in the same digit, yes-rate for non-example targets containing d minus yes-rate for non-example targets not containing d; observed above model means people generalise "contains digit d" more than the model's (family-weighted) contained-digit rules allow, observed below means less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    info = {}
    for s in uniq:
        ex = [int(x) for x in s.split(",")]
        if len(ex) < 2 or len(set(e % 10 for e in ex)) == 1:
            continue
        common = set("0123456789")
        for e in ex:
            common &= set(str(e))
        if common:
            info[s] = (set(ex), sorted(common))
    if not info:
        return 0.0
    m = sets.isin(list(info.keys())).to_numpy()
    sub = df[m]
    tstr = sub["target"].astype(int).astype(str).to_numpy()
    tint = sub["target"].astype(int).to_numpy()
    ss = sub["number_set"].astype(str).to_numpy()
    has = np.array([any(d in t for d in info[s][1]) for s, t in zip(ss, tstr)])
    inset = np.array([t in info[s][0] for s, t in zip(ss, tint)])
    r = sub["rating"].to_numpy(dtype=float)
    a = r[has & ~inset]
    b = r[~has & ~inset]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
