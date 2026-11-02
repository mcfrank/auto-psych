# name: same_decade_near_contrast
# description: For non-example targets 1-5 away from the nearest example, yes rate when the target shares that example's tens decade (floor((n-1)/10)) minus when it crosses a decade boundary; observed above the null means people treat decades as category boundaries that the model's boundary-free intervals ignore (below = no such boundary effect, less than model).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ex = {s: np.array([int(x) for x in s.split(",")]) for s in sets.unique()}
    t = df["target"].astype(int).to_numpy()
    near = np.array([ex[s][np.argmin(np.abs(ex[s] - tt))] for s, tt in zip(sets, t)])
    d = np.abs(near - t)
    ok = (d >= 1) & (d <= 5)
    same = ok & ((near - 1) // 10 == (t - 1) // 10)
    cross = ok & ~same
    y = df["rating"].to_numpy().astype(float)
    if same.sum() == 0 or cross.sum() == 0:
        return 0.0
    return float(y[same].mean() - y[cross].mean())
