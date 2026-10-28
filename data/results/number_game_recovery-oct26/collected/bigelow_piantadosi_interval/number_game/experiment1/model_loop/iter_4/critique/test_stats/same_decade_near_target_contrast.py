# name: same_decade_near_target_contrast
# description: Among non-example targets 1-5 away from their nearest example, yes-rate when the target shares its tens digit (decade) with some example minus when it shares none; observed above null means people use decade membership that the model (which has no decade concept) under-produces.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"])
    E = np.full((len(uniq), 4), -1000)
    for i, s in enumerate(uniq):
        v = [int(x) for x in str(s).split(",")]
        E[i, : len(v)] = v
    arr = E[codes]
    valid = arr > 0
    n_ex = valid.sum(axis=1)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    is_ex = (arr == t[:, None]).any(axis=1)
    lo = np.where(valid, arr, 1000).min(axis=1)
    hi = np.where(valid, arr, -1000).max(axis=1)
    dist = np.abs(np.where(valid, arr, 10000) - t[:, None]).min(axis=1)
    same = (valid & ((arr // 10) == (t // 10)[:, None])).any(axis=1)
    near = (dist >= 1) & (dist <= 5)
    a, b = r[near & same], r[near & ~same]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
