# name: low_evidence_item_drift
# description: Among (set,target) cells that other responders mostly reject (leave-one-out cell yes-rate < 0.3), yes-rate on late trials (>=300) minus early trials (<150); observed above the null means the late "yes" drift reaches even clearly-bad items (lapse/guessing-like), which a logit criterion shift under-produces at the floor; below means the drift spares them more than the model predicts.
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    y = df["rating"].astype(float)
    s = y.groupby(key).transform("sum").values
    n = y.groupby(key).transform("size").values.astype(float)
    yv = y.values
    ok = n > 1
    loo = np.where(ok, (s - yv) / np.maximum(n - 1, 1), np.nan)
    ti = df["trial_index"].astype(float).values
    sel = ok & (loo < 0.3)
    l, e = sel & (ti >= 300), sel & (ti < 150)
    if l.sum() == 0 or e.sum() == 0:
        return 0.0
    return float(yv[l].mean() - yv[e].mean())
