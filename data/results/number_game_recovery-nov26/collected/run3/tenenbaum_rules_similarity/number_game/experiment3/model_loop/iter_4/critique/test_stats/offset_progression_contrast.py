# name: offset_progression_contrast
# description: In two-example sets with gap d=|a-b| in 3..12 where the examples are not multiples of d, yes rate for unshown targets congruent to the examples mod d minus yes rate for other same-parity targets; observed above the null means people extend arithmetic progressions with an offset (e.g. 3,7 -> 11,15) that the model has no concept for.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    sv = sets.values
    a_mask = np.zeros(len(df), bool)
    b_mask = np.zeros(len(df), bool)
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        if len(ex) != 2:
            continue
        d = abs(ex[0] - ex[1])
        if d < 3 or d > 12 or ex[0] % d == 0:
            continue
        r = sv == s
        cong = (t % d) == (ex[0] % d)
        same_par = (t % 2) == (ex[0] % 2)
        unshown = ~np.isin(t, ex)
        a_mask |= r & cong & unshown
        b_mask |= r & ~cong & same_par
    if a_mask.sum() == 0 or b_mask.sum() == 0:
        return 0.0
    return float(y[a_mask].mean() - y[b_mask].mean())
