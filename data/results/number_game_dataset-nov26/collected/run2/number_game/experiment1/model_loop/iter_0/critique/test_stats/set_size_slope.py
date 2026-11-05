# name: set_size_slope
# description: OLS slope of yes (rating) on the number of examples in the set (1-4); observed less negative than null means humans tighten generalization with more examples less than the size principle in the model implies, more negative means they tighten more.
def test_statistic(df):
    k = (df["number_set"].astype(str).str.count(",") + 1).to_numpy(dtype=float)
    y = df["rating"].to_numpy().astype(float)
    kv = k - k.mean()
    den = (kv ** 2).sum()
    return float((kv * (y - y.mean())).sum() / den) if den > 0 else 0.0
