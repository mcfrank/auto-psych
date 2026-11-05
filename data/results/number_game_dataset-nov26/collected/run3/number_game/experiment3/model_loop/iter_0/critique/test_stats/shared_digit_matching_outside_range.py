# name: shared_digit_matching_outside_range
# description: On 3+ example sets whose examples all share a last digit, yes rate for non-example targets with that last digit lying outside the examples' range; observed below null means the model over-extends a shared-digit rule beyond the range (too-crisp ceiling / no lapse).
def test_statistic(df):
    sp = df["number_set"].str.split(",", expand=True)
    v = sp.apply(pd.to_numeric, errors="coerce")
    n = v.notna().sum(axis=1)
    d = v % 10
    dmin, dmax = d.min(axis=1), d.max(axis=1)
    lo, hi = v.min(axis=1), v.max(axis=1)
    t = df["target"].astype(int)
    isex = v.eq(t, axis=0).any(axis=1)
    sel = (n >= 3) & (dmin == dmax) & ((t % 10) == dmin) & (~isex) & ((t < lo) | (t > hi))
    if sel.sum() == 0:
        return 0.0
    return float(df.loc[sel, "rating"].mean())
