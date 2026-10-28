# name: large_gap_interior_yes
# description: On sets of >=2 examples whose largest gap between consecutive sorted examples is >=30, yes-rate to non-example targets inside that gap and >=10 from both of its ends; observed above null means people fill in between distant example clusters (one wide range) more than the model, below null means they treat the clusters as separate more than the model.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.replace(" ", "")
    def gap(v):
        a = sorted(int(z) for z in v)
        if len(a) < 2:
            return (0, 0)
        g = np.diff(a)
        i = int(np.argmax(g))
        return (a[i], a[i + 1])
    u = sets.unique()
    lut = {k: gap(k.split(",")) for k in u}
    lo = sets.map({k: v[0] for k, v in lut.items()}).to_numpy()
    hi = sets.map({k: v[1] for k, v in lut.items()}).to_numpy()
    t = df["target"].astype(int).to_numpy()
    m = (hi - lo >= 30) & (t >= lo + 10) & (t <= hi - 10)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
