# name: high_evidence_item_drift
# description: Among (set,target) cells that other responders mostly endorse (leave-one-out cell yes-rate > 0.7), yes-rate on late trials (>=300) minus early trials (<150); observed below the null means people become less consistent on clearly-good items late (drift is noise/lapses, not a pure criterion relaxation), above means the late "yes" shift is stronger at the ceiling than the model's logit shift allows.
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    y = df["rating"].astype(float)
    s = y.groupby(key).transform("sum").values
    n = y.groupby(key).transform("size").values.astype(float)
    yv = y.values
    ok = n > 1
    loo = np.where(ok, (s - yv) / np.maximum(n - 1, 1), np.nan)
    ti = df["trial_index"].astype(float).values
    sel = ok & (loo > 0.7)
    l, e = sel & (ti >= 300), sel & (ti < 150)
    if l.sum() == 0 or e.sum() == 0:
        return 0.0
    return float(yv[l].mean() - yv[e].mean())
