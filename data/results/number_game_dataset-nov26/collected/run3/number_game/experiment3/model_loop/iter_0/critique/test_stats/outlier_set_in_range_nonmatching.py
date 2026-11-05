# name: outlier_set_in_range_nonmatching
# description: On 3-4 example sets where all but exactly one example share a last digit, yes rate for in-range non-example targets lacking that majority digit; observed below null means the model's dropping of the broken bounded rule over-endorses in-range non-members (observed above = still under-endorsed).
def test_statistic(df):
    sp = df["number_set"].str.split(",", expand=True)
    v = sp.apply(pd.to_numeric, errors="coerce")
    n = v.notna().sum(axis=1)
    d = v % 10
    lo, hi = v.min(axis=1), v.max(axis=1)
    t = df["target"].astype(int)
    isex = v.eq(t, axis=0).any(axis=1)
    best_cnt = pd.Series(0, index=df.index)
    best_dig = pd.Series(-1, index=df.index)
    for k in range(10):
        c = d.eq(k).sum(axis=1)
        upd = c > best_cnt
        best_cnt = best_cnt.where(~upd, c)
        best_dig = best_dig.where(~upd, k)
    sel = (n >= 3) & (best_cnt == n - 1) & (~isex) & (t >= lo) & (t <= hi) & ((t % 10) != best_dig)
    if sel.sum() == 0:
        return 0.0
    return float(df.loc[sel, "rating"].mean())
