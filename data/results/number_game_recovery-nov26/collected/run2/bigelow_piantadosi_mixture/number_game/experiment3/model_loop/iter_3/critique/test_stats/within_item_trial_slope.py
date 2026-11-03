# name: within_item_trial_slope
# description: Correlation between rating and trial_index after subtracting each (set, target) cell's mean of both; observed below null means the same item gets fewer yes answers later in the session than the time-blind model predicts (a true drift, not stimulus order).
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    nd = x.sub(t, axis=0).abs().min(axis=1)
    key = df["number_set"].astype(str) + "|" + t.astype(str)
    ti = df["trial_index"].astype(float)
    rr = r - r.groupby(key).transform("mean")
    tt = ti - ti.groupby(key).transform("mean")
    v = float((rr * tt).sum() / np.sqrt((rr ** 2).sum() * (tt ** 2).sum()))
    return float(0.0 if (v is None or not np.isfinite(v)) else v)
