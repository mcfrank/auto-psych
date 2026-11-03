# name: yes_rate_drift_early_minus_late
# description: Yes rate on trials with trial_index < 150 minus yes rate on trials with trial_index >= 300; observed above null means people grow more conservative over the session than the stationary model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mx = x.max(axis=1)
    mn = x.min(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    ti = df["trial_index"].astype(int)
    a = r[ti < 150].mean()
    b = r[ti >= 300].mean()
    return float(0.0 if (np.isnan(a) or np.isnan(b)) else a - b)
