# name: near_first_minus_near_last_example
# description: Among sets with 2+ examples and non-example targets, yes rate for targets within 5 of the FIRST shown example minus yes rate for targets within 5 of the LAST shown example; observed above null means the first example anchors generalization more than the model (which only uses last-step heading) predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mx = x.max(axis=1)
    mn = x.min(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    first = x[0]
    last = x.ffill(axis=1).iloc[:, -1]
    base = (n >= 2) & ~isin
    nf = base & ((t - first).abs() <= 5) & ((t - last).abs() > 5)
    nl = base & ((t - last).abs() <= 5) & ((t - first).abs() > 5)
    a = r[nf].mean()
    b = r[nl].mean()
    return float(0.0 if (np.isnan(a) or np.isnan(b)) else a - b)
