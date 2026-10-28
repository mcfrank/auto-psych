# name: mid_span_interior_yes
# description: Among sets of 2+ examples, yes-rate for non-example targets strictly inside the example range; observed above null means the model under-produces interval (range-filling) generalisation.
def test_statistic(df):
    s = df["number_set"].astype(str)
    sp = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    lo = sp.min(axis=1).values
    hi = sp.max(axis=1).values
    n = sp.notna().sum(axis=1).values
    t = df["target"].astype(float).values
    isex = (sp.values == t[:, None]).any(axis=1)
    m = (n >= 2) & (t > lo) & (t < hi) & (~isex)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).values[m].mean())
