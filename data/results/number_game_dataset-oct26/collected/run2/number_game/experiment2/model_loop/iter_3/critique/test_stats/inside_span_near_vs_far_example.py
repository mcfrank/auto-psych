# name: inside_span_near_vs_far_example
# description: In 2+-example sets spanning >= 30, among non-example targets strictly inside the span, yes-rate for targets within 5 of the nearest example minus targets more than 12 from every example; observed above the null means people still generalize by distance to individual examples inside the span (gaps are not "filled in" as the model assumes), below means gaps are filled even more uniformly than modelled.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    ex = {s: np.array([int(v) for v in s.split(",")]) for s in sets}
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    near = np.full(len(t), np.nan)
    inside = np.zeros(len(t), bool)
    codes, uniq = pd.factorize(ns)
    for i, s in enumerate(uniq):
        e = ex[s]
        if len(e) < 2 or e.max() - e.min() < 30:
            continue
        m = codes == i
        tt = t[m]
        near[m] = np.abs(tt[:, None] - e[None, :]).min(axis=1)
        inside[m] = (tt > e.min()) & (tt < e.max())
    sel = inside & (near > 0)
    a = sel & (near <= 5)
    b = sel & (near > 12)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
