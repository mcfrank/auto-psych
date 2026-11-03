# name: single_example_near_far_gradient
# description: Single-example sets: yes-rate for targets within 10 of the example minus yes-rate for targets >30 away; observed above null means people's similarity gradient is steeper than the model's.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    d = np.abs(t - ex[:, 0])
    return rate((n == 1) & (d <= 10) & (d > 0)) - rate((n == 1) & (d > 30))
