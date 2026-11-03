# name: within_range_setsize_contrast
# description: Yes rate to non-example targets lying strictly between the min and max example for 4-example sets minus the same for 2-example sets; observed above null means larger sets fill in their range more than the model predicts (weaker size-principle tightening), below means more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").reindex(columns=range(4)).values
    ok = ~np.isnan(ex)
    n = ok.sum(1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    t = df["target"].astype(int).values
    is_ex = np.any(ok & (ex == t[:, None]), axis=1)
    inside = (t > lo) & (t < hi) & ~is_ex
    r = df["rating"].values
    a = inside & (n == 4)
    b = inside & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
