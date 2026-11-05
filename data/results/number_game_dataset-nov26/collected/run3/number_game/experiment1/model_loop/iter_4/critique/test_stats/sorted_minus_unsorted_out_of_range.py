# name: sorted_minus_unsorted_out_of_range
# description: Among sets of >=3 examples, yes rate for non-example targets outside the examples' min-max range in sorted (strictly monotone) sets minus in unsorted sets; the model's sorted-order shift toward range concepts predicts sorted < unsorted, so observed above null means sorted lists do not make people more range-bound (the order shift is an over-general side effect).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    d = ex.diff(axis=1).iloc[:, 1:]
    valid = d.notna()
    inc = ((d > 0) | ~valid).all(axis=1); dec = ((d < 0) | ~valid).all(axis=1)
    srt = (inc | dec) & (n >= 3)
    out = (t < ex.min(axis=1)) | (t > ex.max(axis=1))
    base = (n >= 3) & out
    a = base & srt; b = base & ~srt
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
