# name: above_vs_below_span_asymmetry
# description: Yes rate for targets 1-15 above the set max minus yes rate for targets 1-15 below the set min; model intervals are width-only (symmetric), so observed != null_mean shows a directional (upward vs downward) generalization bias the model misses.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(s, axis=1); mx = np.nanmax(s, axis=1)
    n = np.sum(~np.isnan(s), axis=1)
    up = (t > mx) & (t - mx <= 15)
    dn = (t < mn) & (mn - t <= 15)
    if not up.any() or not dn.any():
        return 0.0
    return float(r[up].mean() - r[dn].mean())
