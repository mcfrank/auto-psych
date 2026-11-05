# name: above_minus_below_range_multi
# description: On sets of 2+ examples, yes rate for non-example targets above the max example minus yes rate for targets below the min example; observed above null means people extrapolate upward more (downward less) than the model's symmetric intervals predict.
def test_statistic(df):
    sp = df["number_set"].str.split(",", expand=True)
    n = sp.notna().sum(axis=1)
    v = sp.apply(pd.to_numeric, errors="coerce")
    lo, hi = v.min(axis=1), v.max(axis=1)
    t = df["target"].astype(int)
    above = (n >= 2) & (t > hi)
    below = (n >= 2) & (t < lo)
    if above.sum() == 0 or below.sum() == 0:
        return 0.0
    r = df["rating"]
    return float(r[above].mean() - r[below].mean())
