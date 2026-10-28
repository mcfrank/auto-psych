# name: far_single_digit_match_multi
# description: Among multi-example sets whose examples do not all share a final digit, yes rate for targets >10 away from every example that share a final digit with at least one example, minus yes rate for targets >10 away sharing no final digit with any example; observed > model means a single example's final digit drives generalization more than the model's recency-weighted, untuned digit attention predicts.
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).reindex(columns=range(4))
    ex = ex.apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    valid = ~np.isnan(ex)
    k = valid.sum(1)
    dig = np.where(valid, np.nan_to_num(ex) % 10, -1)
    nd = np.where(valid, dig, np.nan)
    allsame = (np.nanmax(nd, 1) == np.nanmin(nd, 1))
    d = np.where(valid, np.abs(t[:, None] - ex), np.inf)
    far = d.min(1) > 10
    match = (valid & (dig == (t % 10)[:, None])).any(1)
    base = (k >= 2) & ~allsame & far
    a = base & match
    b = base & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
