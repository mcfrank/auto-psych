# name: mersenne_fermat_target_contrast
# description: In sets whose examples are all of the form 2^k-1 (e.g. 31,3,1,15) or all 2^k+1 (e.g. 33,17,5,9), yes rate for unshown odd targets of the same form minus yes rate for other odd targets; observed above the null means people use "one off a power of 2" concepts the model lacks (model under-produces the contrast).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    minus = {2**k - 1 for k in range(1, 8) if 2**k - 1 <= 100}
    plus = {2**k + 1 for k in range(0, 7) if 2**k + 1 <= 100}
    uniq = sets.unique()
    fam = {}
    for s in uniq:
        ex = [int(v) for v in s.split(",")]
        if len(ex) < 2:
            continue
        if all(e in minus for e in ex):
            fam[s] = (minus, set(ex))
        elif all(e in plus for e in ex):
            fam[s] = (plus, set(ex))
    if not fam:
        return 0.0
    m = sets.isin(list(fam)).values
    inside = np.zeros(len(df), bool)
    shown = np.zeros(len(df), bool)
    sv = sets.values
    for s, (F, ex) in fam.items():
        r = sv == s
        inside |= r & np.isin(t, list(F))
        shown |= r & np.isin(t, list(ex))
    odd = (t % 2 == 1)
    a = m & inside & ~shown & odd
    b = m & ~inside & odd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
