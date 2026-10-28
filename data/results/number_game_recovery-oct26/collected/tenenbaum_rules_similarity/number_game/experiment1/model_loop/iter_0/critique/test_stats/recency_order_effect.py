# name: recency_order_effect
# description: For sets of 2+ examples and non-example targets, yes-rate when the target is strictly closer to the last-shown example than to the first-shown one minus the reverse; the model is order-invariant, so observed above the null means humans show recency (weight the last example more), below means primacy.
def test_statistic(df):
    arr = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = np.sum(~np.isnan(arr), axis=1)
    t = df["target"].astype(float).values
    first = arr[:, 0]
    idx = np.clip(n - 1, 0, arr.shape[1] - 1)
    last = arr[np.arange(len(arr)), idx]
    dmin = np.nanmin(np.abs(arr - t[:, None]), axis=1)
    df_ = np.abs(first - t)
    dl = np.abs(last - t)
    ok = (n >= 2) & (dmin > 0)
    r = df["rating"].values.astype(float)
    a = ok & (dl < df_)
    b = ok & (df_ < dl)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
