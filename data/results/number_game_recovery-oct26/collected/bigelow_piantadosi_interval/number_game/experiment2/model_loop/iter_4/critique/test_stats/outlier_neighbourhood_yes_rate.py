# name: outlier_neighbourhood_yes_rate
# description: In sets of 3-4 examples where all but one share a ones digit (one "outlier" example), the yes rate to non-example targets within 5 of the outlier that do not share the majority ones digit; observed above the null means people generalise around a rule-violating example more than the model (which treats it as noise / a brief switch), below means less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    od, dg = {}, {}
    for s in sets.unique():
        e = [int(x) for x in s.split(",")]
        od[s], dg[s] = -1000, -1
        if len(e) >= 3:
            d = [x % 10 for x in e]
            vals, cnt = np.unique(d, return_counts=True)
            if cnt.max() == len(e) - 1:
                m = vals[cnt.argmax()]
                od[s] = [x for x in e if x % 10 != m][0]
                dg[s] = int(m)
    o = sets.map(od).to_numpy()
    m = sets.map(dg).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    sel = (o > 0) & (np.abs(t - o) <= 5) & (t != o) & (t % 10 != m)
    if sel.sum() == 0:
        return 0.0
    return float(r[sel].mean())
