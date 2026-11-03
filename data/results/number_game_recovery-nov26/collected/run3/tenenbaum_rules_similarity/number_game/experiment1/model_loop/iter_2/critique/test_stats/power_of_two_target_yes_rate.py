# name: power_of_two_target_yes_rate
# description: Among sets whose examples are all powers of 2, the yes-rate for non-example targets that are themselves powers of 2 (e.g. 32 after "8,16", 8 after "64,4"); observed below model means people endorse the powers-of-2 concept less (the model's powers family prior / size principle is too strong), above means more.
def test_statistic(df):
    pw = {2, 4, 8, 16, 32, 64}
    sets = df["number_set"].astype(str)
    keep, exsets = [], {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        if all(e in pw for e in ex):
            keep.append(s)
            exsets[s] = set(ex)
    if not keep:
        return 0.0
    sub = df[sets.isin(keep).to_numpy()]
    t = sub["target"].astype(int).to_numpy()
    ss = sub["number_set"].astype(str).to_numpy()
    inset = np.array([x in exsets[s] for s, x in zip(ss, t)])
    r = sub["rating"].to_numpy(dtype=float)
    sel = np.isin(t, list(pw)) & ~inset
    if sel.sum() == 0:
        return 0.0
    return float(r[sel].mean())
