# name: final_digit_with_outlier_yes
# description: In sets of >=3 examples where all but exactly one share a final digit, yes rate on non-example targets ending in that majority digit; observed > null means people tolerate an off-rule example more than the model's fixed 0.1 noise allows (observed < null: less).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    feat = {}
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        d = -1
        if len(ex) >= 3:
            ds = [v % 10 for v in ex]
            for k in set(ds):
                if ds.count(k) == len(ex) - 1:
                    d = k
        feat[s] = (d, ex)
    dig = sets.map(lambda s: feat[s][0]).to_numpy(int)
    isex = np.array([ti in feat[s][1] for s, ti in zip(sets, t)])
    m = (dig >= 0) & ~isex & (t % 10 == dig)
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
