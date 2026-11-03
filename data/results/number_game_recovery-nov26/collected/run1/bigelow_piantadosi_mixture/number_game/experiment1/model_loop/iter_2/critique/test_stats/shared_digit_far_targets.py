# name: shared_digit_far_targets
# description: For targets more than 15 beyond the examples' range, yes-rate when the target shares a last digit with some example minus when it shares none; observed > null means people use surface digit similarity more than the model's rules capture.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    d = np.maximum(np.nanmin(ex, axis=1) - t, t - np.nanmax(ex, axis=1))
    share = np.any(np.mod(ex, 10) == np.mod(t, 10)[:, None], axis=1)
    far = d > 15
    m1 = far & share; m0 = far & ~share
    a = r[m1].mean() if m1.any() else 0.0; b = r[m0].mean() if m0.any() else 0.0
    return float(a - b)
