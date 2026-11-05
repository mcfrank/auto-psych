# name: proximity_near_minus_far
# description: Among non-example targets, yes rate for targets within 3 of the nearest example minus yes rate for targets more than 20 away; observed above null means people show stronger graded numeric-similarity generalization than the model, below null means weaker.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    inset = ex.eq(t, axis=0).any(axis=1)
    def m(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    dist = ex.sub(t, axis=0).abs().min(axis=1)
    return m((~inset) & (dist <= 3)) - m((~inset) & (dist > 20))
