# name: far_shared_last_digit_effect
# description: Among targets more than 10 from every example, yes rate when the target shares its final digit with at least one example minus yes rate when it shares none; observed above null_mean means people generalize on partial digit similarity that the model (which only uses all-examples digit rules) under-produces.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    d = np.nanmin(np.abs(s - t[:, None]), axis=1)
    share = np.nansum((s % 10) == (t[:, None] % 10), axis=1) > 0
    far = d > 10
    y = df["rating"].astype(float).to_numpy()
    a = far & share; b = far & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
