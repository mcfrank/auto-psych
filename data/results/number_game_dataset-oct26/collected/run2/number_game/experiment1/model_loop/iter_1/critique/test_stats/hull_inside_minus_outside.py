# name: hull_inside_minus_outside
# description: Among sets with >=2 examples spanning a range of >=10, the yes rate for targets strictly inside [min,max] of the examples minus targets outside it; observed > model means people use an interval/range rule with a sharper inside-vs-outside boundary than graded magnitude similarity predicts.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    lo, hi = {}, {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        lo[s], hi[s] = min(ex), max(ex)
    L = sets.map(lo).to_numpy(); H = sets.map(hi).to_numpy()
    k = (sets.str.count(",") + 1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    m = (k >= 2) & (H - L >= 10)
    ins = m & (t > L) & (t < H)
    out = m & ((t < L) | (t > H))
    if ins.sum() == 0 or out.sum() == 0:
        return 0.0
    return float(r[ins].mean() - r[out].mean())
