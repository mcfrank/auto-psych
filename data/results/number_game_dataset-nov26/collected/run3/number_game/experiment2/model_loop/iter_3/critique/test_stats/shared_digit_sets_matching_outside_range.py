# name: shared_digit_sets_matching_outside_range
# description: For 3+-example sets whose examples all share a last digit, yes rate on non-example targets with that same last digit lying outside [min,max] (48,78,38,98->8,18; 64,94,84->14); observed above null means people extend a digit rule beyond the examples' range more than the model's range-bounded rules allow (below = model over-extends).
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    sub = df[n >= 3]
    parts = sub["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    ld = parts % 10
    k = parts.notna().sum(axis=1)
    same = (ld.nunique(axis=1) == 1) & (k >= 3)
    d0 = ld[0]
    t = sub["target"].astype(int)
    is_ex = parts.eq(t, axis=0).any(axis=1)
    lo = parts.min(axis=1); hi = parts.max(axis=1)
    m = same & (t % 10 == d0) & ~is_ex & ((t < lo) | (t > hi))
    if m.sum() == 0:
        return 0.0
    return float(sub["rating"][m].mean())
