# name: upward_minus_downward_near
# description: For non-example targets within 15 of their nearest example, yes rate when the target is above that example minus when it is below; observed > null means people generalise more upward than the model's roughly symmetric intervals predict, observed < null means more downward.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    diff = ex.rsub(t, axis=0)  # target - example
    ad = diff.abs()
    mn = ad.min(axis=1)
    # signed diff to nearest example (ties: prefer any above)
    near_up = (diff.gt(0) & ad.eq(mn, axis=0)).any(axis=1)
    near_dn = (diff.lt(0) & ad.eq(mn, axis=0)).any(axis=1)
    ok = (mn > 0) & (mn <= 15)
    r = df["rating"].astype(float)
    up = ok & near_up & ~near_dn
    dn = ok & near_dn & ~near_up
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
