# name: interior_sets_above_minus_below
# description: For sets lying within 20-80 (away from the 1/100 edges), yes rate for non-example targets 1-15 above the largest example minus targets 1-15 below the smallest; the model's interval prior is symmetric, so observed below null_mean means people extend generalization downward more than upward relative to the model, above means the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n_ex = valid.sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    out_d = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    base = (lo >= 20) & (hi <= 80) & (~member) & (out_d >= 1) & (out_d <= 15)
    a = base & (t > hi); b = base & (t < lo)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
