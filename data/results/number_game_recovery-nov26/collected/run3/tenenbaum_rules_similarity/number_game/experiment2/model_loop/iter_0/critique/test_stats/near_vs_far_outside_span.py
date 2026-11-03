# name: near_vs_far_outside_span
# description: Among targets outside [min,max] of a multi-example set, yes rate for targets 1-10 away from the nearest edge minus yes rate for targets more than 25 away; observed > null means human generalisation falls off with distance beyond the examples more steeply than the model's interval hypotheses produce (observed < null: flatter).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    ok = n >= 2
    r = df["rating"].to_numpy().astype(float)
    a = ok & (dist >= 1) & (dist <= 10); b = ok & (dist > 25)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
