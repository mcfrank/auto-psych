# name: upward_minus_downward_extension
# description: Among non-example targets outside the example range and within 15 of it, yes rate for targets above the max example minus targets below the min example; observed above null means people extend generalization upward more than the model's symmetric interval hypotheses predict, below null means downward more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    lo = ex.min(axis=1); hi = ex.max(axis=1)
    up = (t > hi) & (t - hi <= 15)
    dn = (t < lo) & (lo - t <= 15)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
