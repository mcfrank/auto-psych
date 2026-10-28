# name: same_decade_near_gap
# description: Among targets 1-9 away from their nearest example, the yes rate when the target shares the tens digit (decade) with that nearest example minus when it does not; observed > model means people use a decade/tens-digit feature the model lacks (it codes only magnitude, final digit, parity, divisibility).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ex = sets.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    d = np.abs(ex - t[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    near = ex[np.arange(len(t)), j]
    dmin = d[np.arange(len(t)), j]
    r = df["rating"].astype(float).to_numpy()
    dec = lambda v: np.minimum(v // 10, 9)  # 100 grouped with 90s
    m = (dmin >= 1) & (dmin <= 9)
    same = dec(t) == dec(near)
    a, b = m & same, m & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
