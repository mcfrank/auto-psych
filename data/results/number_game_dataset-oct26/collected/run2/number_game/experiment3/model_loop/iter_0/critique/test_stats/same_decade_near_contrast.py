# name: same_decade_near_contrast
# description: Among targets outside the example set lying within 1-10 of their nearest example, yes-rate when the target shares that example's tens decade minus when it does not; observed > null means people use decade membership (which the model, with only size and final digit, lacks) and the model under-produces this contrast.
def test_statistic(df):
    g = df.groupby(["number_set", "target"], sort=False)["rating"].mean().reset_index()
    same, diff = [], []
    for s, t, r in zip(g["number_set"], g["target"].astype(int), g["rating"]):
        ex = [int(x) for x in str(s).split(",")]
        if t in ex:
            continue
        x = min(ex, key=lambda e: abs(e - t))
        if abs(x - t) > 10:
            continue
        (same if min(x, 99) // 10 == min(t, 99) // 10 else diff).append(r)
    if not same or not diff:
        return 0.0
    return float(np.mean(same) - np.mean(diff))
