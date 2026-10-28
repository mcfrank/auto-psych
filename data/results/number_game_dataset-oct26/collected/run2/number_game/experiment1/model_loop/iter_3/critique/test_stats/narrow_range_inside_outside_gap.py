# name: narrow_range_inside_outside_gap
# description: Among multi-example sets spanning <= 25 (max-min), yes rate for targets strictly inside [min,max] (not in the set) minus yes rate for targets within 15 outside the range; observed > model means people treat a tight cluster as a bounded interval rule, generalizing inside but not past the edges more than graded magnitude similarity predicts.
def test_statistic(df):
    s = df["number_set"].astype(str)
    lo, hi = {}, {}
    for u in s.unique():
        v = [int(z) for z in u.split(",")]
        if len(v) >= 2 and max(v) - min(v) <= 25:
            lo[u], hi[u] = min(v), max(v)
        else:
            lo[u], hi[u] = -1000, -1000
    L = s.map(lo).values.astype(int)
    H = s.map(hi).values.astype(int)
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    m = L > -1000
    inside = m & (t > L) & (t < H)
    outside = m & (((t < L) & (t >= L - 15)) | ((t > H) & (t <= H + 15)))
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(y[inside].mean() - y[outside].mean())
