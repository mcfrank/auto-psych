# name: central_set_low_vs_high_target
# description: For sets whose example mean lies in 35-65, yes rate for small targets (<=20) minus yes rate for large targets (>=81); observed above null means people extend generalization toward small numbers more than toward large numbers relative to the model (a magnitude asymmetry the model lacks), below means the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    mu = np.nanmean(ex, axis=1)
    c = (mu >= 35) & (mu <= 65)
    a, b = c & (t <= 20), c & (t >= 81)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
