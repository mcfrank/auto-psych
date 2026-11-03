# name: magnitude_outlier_neighbourhood
# description: For sets of >=3 examples with one magnitude outlier (an end example whose gap to its neighbour is >=20 and >=2x every other gap, e.g. 100 in 31,19,49,100), yes rate on non-example targets within 5 of the outlier minus targets within 5 of the other examples; observed < null means people discount the isolated example as a stray more than the model's 10% stray rate allows, observed > null means they generalise around it more.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    out = {}
    for s in pd.unique(sets):
        e = sorted(int(x) for x in s.split(","))
        out[s] = (-999, -999, -999)
        if len(e) < 3:
            continue
        g = np.diff(e)
        for idx, gi, rest in ((0, g[0], g[1:]), (len(e) - 1, g[-1], g[:-1])):
            if gi >= 20 and gi >= 2 * rest.max():
                others = [x for j, x in enumerate(e) if j != idx]
                out[s] = (e[idx], min(others), max(others))
    o = pd.Series(sets).map(lambda s: out[s][0]).values
    lo = pd.Series(sets).map(lambda s: out[s][1]).values
    hi = pd.Series(sets).map(lambda s: out[s][2]).values
    valid = o > -999
    isex = np.array([str(x) in s.split(",") for x, s in zip(t, sets)])
    near_o = valid & ~isex & (np.abs(t - o) <= 5)
    near_r = valid & ~isex & ~near_o & (t >= lo - 5) & (t <= hi + 5)
    if near_o.sum() == 0 or near_r.sum() == 0:
        return 0.0
    return float(r[near_o].mean() - r[near_r].mean())
