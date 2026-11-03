# name: odd_rule_conjunction_contrast
# description: In sets of >=2 all-odd examples that are all multiples of some m in {3,5,7,9} or all squares, yes rate for unshown odd members of that rule minus yes rate for even members of it; observed above the null means people infer the conjunction "odd AND rule" more sharply than the model's mixture of separate rules (model under-produces the contrast).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    sv = sets.values
    sq = {n * n for n in range(1, 11)}
    a_mask = np.zeros(len(df), bool)
    b_mask = np.zeros(len(df), bool)
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        if len(ex) < 2 or any(e % 2 == 0 for e in ex):
            continue
        r = sv == s
        member = np.zeros(len(df), bool)
        for m in (3, 5, 7, 9):
            if all(e % m == 0 for e in ex):
                member |= (t % m == 0)
        if all(e in sq for e in ex):
            member |= np.isin(t, list(sq))
        if not member.any():
            continue
        unshown = ~np.isin(t, ex)
        a_mask |= r & member & unshown & (t % 2 == 1)
        b_mask |= r & member & (t % 2 == 0)
    if a_mask.sum() == 0 or b_mask.sum() == 0:
        return 0.0
    return float(y[a_mask].mean() - y[b_mask].mean())
