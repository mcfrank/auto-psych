# name: ratio_vs_magnitude_fixed_gap
# description: Among out-of-span targets at a linear gap of 6-20 from the nearest span edge, yes-rate when |log(target/edge)| < 0.25 minus when > 0.5 (at fixed linear gap this contrasts large vs small magnitudes); observed > null means people judge size by ratio (log-compressed), which the model's linear size distance under-produces.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    exv = ex.to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(exv, axis=1); mx = np.nanmax(exv, axis=1)
    edge = np.where(t < mn, mn, mx)
    gap = np.maximum(np.maximum(mn - t, t - mx), 0)
    band = (gap >= 6) & (gap <= 20)
    lr = np.abs(np.log(np.maximum(t, 1) / np.maximum(edge, 1)))
    a = band & (lr < 0.25)
    b = band & (lr > 0.5)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
