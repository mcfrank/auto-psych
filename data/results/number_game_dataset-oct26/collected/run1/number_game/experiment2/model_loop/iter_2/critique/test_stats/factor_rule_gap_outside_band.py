# name: factor_rule_gap_outside_band
# description: In sets with common factor >=3, among non-example targets outside the decade band sharing no final digit with any example, yes rate for multiples of the common factor minus non-multiples; t_observed above null_mean means people apply the factor rule regardless of magnitude more strongly than the model's additive logit boost allows (below = weaker).
def test_statistic(df):
    from functools import reduce
    t = df["target"].astype(int).values
    sets = df["number_set"].astype(str).values
    grp = np.full(len(t), -1)
    for s in np.unique(sets):
        xs = np.array([int(v) for v in s.split(",")])
        g = reduce(math.gcd, [int(v) for v in xs])
        if g < 3:
            continue
        lo, hi = (xs // 10).min(), (xs // 10).max()
        m = sets == s
        tt = t[m]
        dig = ((tt[:, None] % 10) == (xs[None, :] % 10)).any(1)
        outb = (tt // 10 < lo) | (tt // 10 > hi)
        ok = outb & ~dig
        grp[m] = np.where(ok, (tt % g == 0).astype(int), -1)
    r = df["rating"].astype(float).values
    a = grp == 1
    b = grp == 0
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
