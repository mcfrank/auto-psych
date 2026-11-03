# name: late_session_far_generalization_shift
# description: Yes-rate on targets more than 15 away from the nearest example in the second half of each participant's session (trial_index >= half of that participant's max) minus the first half; the model is stationary, so observed above null means people generalize more broadly as the session goes on, below null means they become stricter.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[1, (np.diff(ti) <= 0).astype(int)])
    mx = pd.Series(ti).groupby(pid).transform("max").to_numpy()
    late = ti >= mx / 2.0
    far = d > 15
    r = df["rating"].to_numpy(dtype=float)
    a = far & late
    b = far & ~late
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
