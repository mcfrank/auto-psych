# name: inside_vs_outside_range_gap
# description: Among sets with >=2 examples and targets not in the set, yes-rate for targets inside [min,max] of the examples minus yes-rate for targets outside; observed > null means the model under-produces the interval-like "fill in the range, reject beyond it" contrast.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1)
    t = df["target"].astype(int)
    mn, mx = s.min(axis=1), s.max(axis=1)
    member = s.eq(t, axis=0).any(axis=1)
    ok = (n >= 2) & ~member
    inside = ok & (t >= mn) & (t <= mx)
    outside = ok & ((t < mn) | (t > mx))
    r = df["rating"].astype(float)
    a = r[inside].mean() if inside.any() else 0.5
    b = r[outside].mean() if outside.any() else 0.5
    return float(a - b)
