# name: round_target_minus_neighbors
# description: Yes rate for non-example targets that are multiples of 10 minus yes rate for non-example targets ending in 1 or 9; observed above null means people favour round-number targets more than the model predicts.
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
    rnd = (t % 10 == 0)
    nb = (t % 10 == 1) | (t % 10 == 9)
    v = r[rnd & ~isin].mean() - r[nb & ~isin].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
