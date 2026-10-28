# name: outside_range_near_minus_far
# description: Among non-example targets lying outside the [min, max] range of a 2+-example set, yes-rate for targets within 10 of the range minus yes-rate for targets more than 25 beyond it; observed > model means people's generalization falls off with distance outside the examples more steeply than the model's interval/rule mixture predicts, observed < model means a flatter falloff.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = (~np.isnan(ex)).sum(axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    tgt = df["target"].astype(float).values
    dist = np.maximum(lo - tgt, tgt - hi)
    valid = (n >= 2) & (dist > 0)
    near = valid & (dist <= 10)
    far = valid & (dist > 25)
    r = df["rating"].astype(float).values
    if near.sum() == 0 or far.sum() == 0:
        return 0.0
    return float(r[near].mean() - r[far].mean())
