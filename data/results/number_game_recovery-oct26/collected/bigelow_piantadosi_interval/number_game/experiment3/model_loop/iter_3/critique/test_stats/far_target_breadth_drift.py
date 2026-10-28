# name: far_target_breadth_drift
# description: Yes rate on non-example targets more than 20 from every example in the second half of the session (trial_index above median) minus the first half; observed below null_mean means people's distant generalization narrows over the session more than the stationary model predicts, above means it broadens.
def test_statistic(df):
    u = df["number_set"].unique()
    ex = {s: np.array([int(x) for x in str(s).split(",")]) for s in u}
    a = np.arange(101)
    dist = np.stack([np.abs(a[:, None] - ex[s][None, :]).min(axis=1) for s in u])
    idx = pd.Series(range(len(u)), index=u)[df["number_set"]].to_numpy()
    t = df["target"].astype(int).to_numpy()
    far = dist[idx, t] > 20
    ti = df["trial_index"].to_numpy()
    late = ti > np.median(ti)
    r = df["rating"].astype(float).to_numpy()
    a1 = far & late
    a0 = far & ~late
    if a1.sum() == 0 or a0.sum() == 0:
        return 0.0
    return float(r[a1].mean() - r[a0].mean())
