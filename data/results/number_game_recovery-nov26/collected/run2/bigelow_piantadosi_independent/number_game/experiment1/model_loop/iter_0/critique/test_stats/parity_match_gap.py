# name: parity_match_gap
# description: Among sets whose examples share one parity (all even or all odd, >=2 examples), yes rate for targets matching that parity minus non-matching; t_observed above null_mean means people weight parity more than the model.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    codes, uniq = pd.factorize(sets)
    par = np.full(len(uniq), -1)
    for i, k in enumerate(uniq):
        a = np.array([int(x) for x in k.split(",")]) % 2
        if len(a) >= 2 and (a == a[0]).all():
            par[i] = a[0]
    p = par[codes]
    tp = df["target"].astype(int).values % 2
    r = df["rating"].values.astype(float)
    ok = p >= 0
    mt, mn = ok & (tp == p), ok & (tp != p)
    if mt.sum() == 0 or mn.sum() == 0:
        return 0.0
    return float(r[mt].mean() - r[mn].mean())
