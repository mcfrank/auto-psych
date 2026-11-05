# name: opposite_parity_multiples_two_examples
# description: For 2-example sets whose examples share parity and are both multiples of some k in 3..12 (largest such k), yes rate of non-example targets that are multiples of k with the opposite parity (e.g. even multiples of 7 for 7,63); observed above null means the model's parity-restricted concepts reject opposite-parity rule members too firmly from only two examples, below null means people reject them even more.

def test_statistic(df):
    d = df[["number_set", "target", "rating"]].copy()
    d["target"] = d["target"].astype(int)
    vals = []
    for s, g in d.groupby("number_set"):
        ex = [int(x) for x in s.split(",")]
        if len(ex) != 2 or ex[0] % 2 != ex[1] % 2:
            continue
        ks = [k for k in range(3, 13) if all(e % k == 0 for e in ex)]
        if not ks:
            continue
        k = max(ks)
        t = g["target"]
        m = (t % k == 0) & (t % 2 != ex[0] % 2) & ~t.isin(ex)
        vals.append(g.loc[m, "rating"].to_numpy())
    v = np.concatenate(vals) if vals else np.array([])
    return float(v.mean()) if len(v) else 0.0
