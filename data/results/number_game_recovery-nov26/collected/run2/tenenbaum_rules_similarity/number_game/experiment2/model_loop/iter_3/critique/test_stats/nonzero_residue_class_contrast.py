# name: nonzero_residue_class_contrast
# description: Among 2+-example sets whose examples all share a nonzero residue r mod k for some k in {3,4,6,7,8,9,11,12} (largest such k; e.g. odd multiples of 3 = 3 mod 6), yes rate for non-example targets with the same residue minus targets with a different residue; observed above the null means people infer offset/intersection concepts ("odd multiples of 3") the model's zero-residue multiples rules lack (below = less than the model).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        kk, rr = 0, -1
        if len(ex) >= 2:
            for k in [12, 11, 9, 8, 7, 6, 4, 3]:
                res = {x % k for x in ex}
                if len(res) == 1 and list(res)[0] != 0:
                    kk, rr = k, list(res)[0]
                    break
        info[s] = (kk, rr, ",".join(str(x) for x in ex))
    k = sets.map(lambda s: info[s][0]).to_numpy()
    r = sets.map(lambda s: info[s][1]).to_numpy()
    t = df["target"].astype(int).to_numpy()
    inset = np.array([str(tt) in s.split(",") for tt, s in zip(t, sets)])
    ok = (k > 0) & ~inset
    if ok.sum() == 0:
        return 0.0
    same = ok & (t % np.maximum(k, 1) == r)
    diff = ok & ~same
    y = df["rating"].to_numpy().astype(float)
    if same.sum() == 0 or diff.sum() == 0:
        return 0.0
    return float(y[same].mean() - y[diff].mean())
