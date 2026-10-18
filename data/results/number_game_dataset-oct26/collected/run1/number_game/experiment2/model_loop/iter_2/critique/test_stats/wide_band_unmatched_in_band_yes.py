# name: wide_band_unmatched_in_band_yes
# description: In sets whose examples span >=5 decades, yes rate for in-band non-example targets sharing no final digit and not a common-factor multiple; t_observed below null_mean means the model over-extends full leading-digit familiarity across wide bands (people do not treat a wide empty band as familiar), above means it under-produces it.
def test_statistic(df):
    from functools import reduce
    t = df["target"].astype(int).values
    sets = df["number_set"].astype(str).values
    sel = np.zeros(len(t), bool)
    for s in np.unique(sets):
        xs = np.array([int(v) for v in s.split(",")])
        lo, hi = (xs // 10).min(), (xs // 10).max()
        if hi - lo < 5:
            continue
        g = reduce(math.gcd, [int(v) for v in xs])
        m = sets == s
        tt = t[m]
        dig = ((tt[:, None] % 10) == (xs[None, :] % 10)).any(1)
        isex = (tt[:, None] == xs[None, :]).any(1)
        cf = (g >= 3) & (tt % max(g, 1) == 0)
        inb = (tt // 10 >= lo) & (tt // 10 <= hi)
        sel[m] = inb & ~dig & ~isex & ~cf
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).values[sel].mean())
