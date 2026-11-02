# name: overshoot_by_set_size_gap
# description: Yes rate on targets just above the largest example (max < target <= 1.5*max) for 1-example sets minus for 3-4-example sets; observed above null means the ceiling should tighten more with the number of examples than the model's set-size-independent ratio ceiling allows (below null = less).
def test_statistic(df):
    info = {}
    for s in df["number_set"].astype(str).unique():
        ex = [int(x) for x in s.split(",")]
        info[s] = (len(ex), max(ex))
    ns = df["number_set"].astype(str).values
    k = np.array([info[s][0] for s in ns])
    mx = np.array([info[s][1] for s in ns]).astype(float)
    tg = df["target"].astype(int).values
    over = (tg > mx) & (tg <= 1.5 * mx)
    r = df["rating"].values.astype(float)
    a = over & (k == 1)
    b = over & (k >= 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
