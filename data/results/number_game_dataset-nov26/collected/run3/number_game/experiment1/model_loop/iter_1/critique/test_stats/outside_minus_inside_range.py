# name: outside_minus_inside_range
# description: Among non-example targets, yes rate for targets outside the [min, max] range of the examples minus targets inside that range; observed above null means people extend beyond the example range more than the model's interval hypotheses allow, below null means they confine generalization to the range more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    inset = ex.eq(t, axis=0).any(axis=1)
    def m(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    lo = ex.min(axis=1); hi = ex.max(axis=1)
    inside = (t >= lo) & (t <= hi)
    return m((~inset) & (~inside)) - m((~inset) & inside)
