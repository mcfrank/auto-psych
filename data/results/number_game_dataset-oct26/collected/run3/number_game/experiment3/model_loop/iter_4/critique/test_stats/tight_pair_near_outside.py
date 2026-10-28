# name: tight_pair_near_outside
# description: For 2-example sets with range <= 10, yes-rate for targets outside [min,max] but within 10 of the nearest example; observed > null means people extend a tight pair to nearby numbers more than the model's size-principle intervals predict (observed < null: they generalize beyond a tight pair less than the model).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = (~np.isnan(ex)).sum(axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    dist = np.maximum(lo - t, t - hi)
    mask = (n == 2) & (hi - lo <= 10) & (dist >= 1) & (dist <= 10)
    return float(r[mask].mean()) if mask.any() else 0.0
