# name: tens_digit_match_outside_range
# description: Among non-example targets outside the [min,max] example range, yes rate for targets sharing a tens digit (decade) with some example minus those that do not; observed above null means people generalize by decade similarity beyond what the model's intervals/rules predict, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    inset = ex.eq(t, axis=0).any(axis=1)
    outside = (t < ex.min(axis=1)) | (t > ex.max(axis=1))
    dec = (ex // 10).where(ex < 100, 9)
    tdec = (t // 10).where(t < 100, 9)
    share = dec.eq(tdec, axis=0).any(axis=1)
    base = outside & (~inset)
    a = base & share; b = base & (~share)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
