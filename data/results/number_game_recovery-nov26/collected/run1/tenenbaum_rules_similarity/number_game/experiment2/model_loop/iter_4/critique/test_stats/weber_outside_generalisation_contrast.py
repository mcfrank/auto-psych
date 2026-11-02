# name: weber_outside_generalisation_contrast
# description: Yes rate for non-member targets 1-15 outside the span of sets lying entirely at or above 50, minus the same rate for sets lying entirely at or below 35; observed above null means people generalise further (in absolute distance) for large numbers than the model's magnitude-independent ranges allow (Weber-like scaling), below means the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    d = np.where(t > hi, t - hi, np.where(t < lo, lo - t, 0))
    near = (d >= 1) & (d <= 15) & ~member
    a = near & (lo >= 50); b = near & (hi <= 35)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
