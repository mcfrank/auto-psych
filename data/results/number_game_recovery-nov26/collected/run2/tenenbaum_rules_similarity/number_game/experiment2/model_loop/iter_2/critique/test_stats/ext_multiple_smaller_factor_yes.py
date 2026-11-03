# name: ext_multiple_smaller_factor_yes
# description: Among sets of 2+ examples that are all multiples of some m in 13..50 (largest such m), yes rate for non-example targets that are not multiples of m but are multiples of a familiar factor f in 3..12 that divides every example; observed above the null means the model's extended multiples-of-13..50 rules over-suppress generalization to the smaller familiar factor (below = people suppress even more).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        if len(ex) < 2:
            continue
        ms = [m for m in range(13, 51) if all(e % m == 0 for e in ex)]
        if not ms:
            continue
        fs = [f for f in range(3, 13) if f != 10 and all(e % f == 0 for e in ex)]
        if not fs:
            continue
        info[s] = (max(ms), fs, set(ex))
    if not info:
        return 0.0
    sub = df[sets.isin(list(info))]
    t = sub["target"].astype(int).to_numpy()
    ss = sub["number_set"].astype(str).to_numpy()
    mask = np.zeros(len(sub), dtype=bool)
    for s, (m, fs, exs) in info.items():
        idx = ss == s
        tt = t[idx]
        ok = (tt % m != 0) & np.isin(tt, list(exs), invert=True)
        fac = np.zeros(len(tt), dtype=bool)
        for f in fs:
            fac |= (tt % f == 0)
        mask[idx] = ok & fac
    if mask.sum() == 0:
        return 0.0
    return float(sub["rating"].to_numpy()[mask].mean())
