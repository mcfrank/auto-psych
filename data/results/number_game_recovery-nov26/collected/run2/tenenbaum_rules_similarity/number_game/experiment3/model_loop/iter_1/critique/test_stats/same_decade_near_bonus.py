# name: same_decade_near_bonus
# description: Among non-example targets 1-9 away from the nearest example and sharing no final digit with any example, yes-rate when the target shares a tens digit (decade) with some example minus yes-rate when it does not; observed above null means the model under-produces a decade-sharing bonus.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    ok = ~np.isnan(ex)
    diff = np.where(ok, np.abs(ex - t[:, None]), np.inf)
    dist = diff.min(axis=1)
    last = np.where(ok, (np.nan_to_num(ex) % 10) == (t % 10)[:, None], False).any(axis=1)
    tens = np.where(ok, (np.nan_to_num(ex) // 10) == (t // 10)[:, None], False).any(axis=1)
    m = (dist >= 1) & (dist <= 9) & ~last
    a, b = m & tens, m & ~tens
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
