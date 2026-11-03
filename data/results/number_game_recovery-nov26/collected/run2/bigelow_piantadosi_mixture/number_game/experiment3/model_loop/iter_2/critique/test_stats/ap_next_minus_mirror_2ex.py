# name: ap_next_minus_mirror_2ex
# description: For 2-example sets a,b, yes rate on the next sequence term 2b-a minus on the mirrored term 2a-b (equally far from the nearer example); observed above null means people continue a two-step sequence in its shown direction more than the model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    dist = x.sub(t, axis=0).abs()
    nd = dist.min(axis=1)
    a = x[0]
    b = x[1] if x.shape[1] > 1 else a
    two = (n == 2) & (a != b)
    v = r[two & (t == 2 * b - a)].mean() - r[two & (t == 2 * a - b)].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
