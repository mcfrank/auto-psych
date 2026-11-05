# name: small_vs_large_set_yes_rate_diff
# description: Overall yes-rate on sets with 1-2 examples minus yes-rate on sets with 3-4 examples; observed > null means people generalize more broadly from few examples (weaker size principle) than the model, observed < null the reverse.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",").values + 1
    r = df["rating"].values.astype(float)
    a = n <= 2
    b = n >= 3
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
