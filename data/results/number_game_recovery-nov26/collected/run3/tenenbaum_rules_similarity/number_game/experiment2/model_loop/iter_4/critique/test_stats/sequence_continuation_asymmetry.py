# name: sequence_continuation_asymmetry
# description: For sets of >=2 examples, yes rate on the forward arithmetic continuation of the shown order (last + (last - second-last)) minus the backward continuation (first - (second - first)), targets in 1..100; observed > null (null reflects only content, model is order-invariant) means people extrapolate the shown sequence in its presented direction, observed < null means the reverse.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    fw, bw = {}, {}
    for s in pd.unique(sets):
        e = [int(x) for x in s.split(",")]
        if len(e) >= 2:
            fw[s] = e[-1] + (e[-1] - e[-2])
            bw[s] = e[0] - (e[1] - e[0])
        else:
            fw[s] = -999
            bw[s] = -999
    f = pd.Series(sets).map(fw).values
    b = pd.Series(sets).map(bw).values
    a1 = r[t == f]
    a2 = r[t == b]
    if len(a1) == 0 or len(a2) == 0:
        return 0.0
    return float(a1.mean() - a2.mean())
