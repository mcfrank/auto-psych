# name: magnitude_scaled_extension
# description: Yes-rate on targets 1-10 outside the examples' span for sets whose example mean is >= 60, minus the same for sets whose mean is <= 40; observed above null_mean means people extend further around large numbers (log/compressed number line) than the model's uniform-width intervals predict (below = the reverse).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    mu = np.nanmean(ex, axis=1)
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    y = df["rating"].astype(float).to_numpy()
    band = (dist >= 1) & (dist <= 10)
    big = band & (mu >= 60)
    small = band & (mu <= 40)
    if big.sum() == 0 or small.sum() == 0:
        return 0.0
    return float(y[big].mean() - y[small].mean())
