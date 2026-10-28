# name: decade_boundary_effect
# description: Among non-example targets 1-5 away from their nearest example, yes rate when the target shares that example's tens decade (same value of (x-1)//10) minus when it crosses a decade boundary; observed above null_mean means people treat decades as categories beyond what the model's uniform intervals and rules predict, below means less.
def test_statistic(df):
    u = df["number_set"].unique()
    ex = {s: np.array([int(x) for x in str(s).split(",")]) for s in u}
    t = df["target"].astype(int).to_numpy()
    tab = np.zeros((len(u), 101), dtype=int)
    dist = np.zeros((len(u), 101), dtype=int)
    for i, s in enumerate(u):
        a = np.arange(101)
        dd = np.abs(a[:, None] - ex[s][None, :])
        k = dd.argmin(axis=1)
        tab[i] = ex[s][k]
        dist[i] = dd.min(axis=1)
    idx = pd.Series(range(len(u)), index=u)[df["number_set"]].to_numpy()
    near = tab[idx, t]
    d = dist[idx, t]
    keep = (d >= 1) & (d <= 5)
    same = keep & ((t - 1) // 10 == (near - 1) // 10)
    cross = keep & ~same
    r = df["rating"].astype(float).to_numpy()
    if same.sum() == 0 or cross.sum() == 0:
        return 0.0
    return float(r[same].mean() - r[cross].mean())
