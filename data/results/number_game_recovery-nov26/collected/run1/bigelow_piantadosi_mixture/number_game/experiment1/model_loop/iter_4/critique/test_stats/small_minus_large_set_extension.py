# name: small_minus_large_set_extension
# description: Yes-rate for targets 1-15 outside the examples' range in sets whose max <= 40 minus that in sets whose min >= 55; observed > null means people extend small-number sets more (relative to large-number sets) than the model's power-law endpoint prior predicts, observed < null means less.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    lo = parts.min(axis=1).values
    hi = parts.max(axis=1).values
    t = df["target"].astype(float).values
    gap = np.maximum(lo - t, t - hi)
    out = (gap >= 1) & (gap <= 15)
    y = df["rating"].values.astype(float)
    a, b = out & (hi <= 40), out & (lo >= 55)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
