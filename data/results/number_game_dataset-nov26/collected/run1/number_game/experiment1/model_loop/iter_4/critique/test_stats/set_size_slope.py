# name: set_size_slope
# description: OLS slope of trial-level yes response on the number of examples in the set (1-4); observed < null means more examples make people more conservative (size-principle tightening) than the model's summed similarity allows, observed > null means more examples make people more liberal than predicted.
def test_statistic(df):
    n = (df["number_set"].str.count(",") + 1).astype(float).values
    y = df["rating"].astype(float).values
    nc = n - n.mean()
    den = float((nc ** 2).sum())
    if den == 0:
        return 0.0
    return float((nc * (y - y.mean())).sum() / den)
