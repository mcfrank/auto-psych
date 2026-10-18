# name: inside_minus_outside_range
# description: For sets with 2-4 examples, yes rate for targets inside [min,max] of the examples minus yes rate for targets outside that range; observed above null means people treat the span of examples as a hypothesis region more strongly than summed exemplar similarity predicts.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = s.notna().sum(axis=1)
    lo = s.min(axis=1)
    hi = s.max(axis=1)
    t = df["target"].astype(float)
    r = df["rating"].astype(float)
    multi = n >= 2
    inside = multi & (t >= lo) & (t <= hi)
    outside = multi & ~((t >= lo) & (t <= hi))
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
