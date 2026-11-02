# name: single_example_same_last_digit_far
# description: Yes-rate for single-example sets on targets >=20 away sharing the example's last digit; observed above null means the model under-produces digit generalisation from one example.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    x = ex[:, 0]
    m = (n == 1) & (np.abs(t - x) >= 20) & (np.mod(t, 10) == np.mod(x, 10))
    return rate(m)
