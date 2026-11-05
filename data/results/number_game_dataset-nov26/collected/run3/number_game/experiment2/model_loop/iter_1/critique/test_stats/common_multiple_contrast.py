# name: common_multiple_contrast
# description: For 2+-example sets whose examples share a common divisor k in 3..12 (largest such k), mean over sets of (yes rate on non-example multiples of k) minus (yes rate on non-multiples); observed below null means people use arithmetic "multiples of k" rules less sharply than the model (above = more sharply).
def test_statistic(df):
    sets = df["number_set"].unique()
    kmap = {}; exs = {}
    for s in sets:
        xs = [int(v) for v in str(s).split(",")]
        if len(xs) < 2:
            continue
        ks = [k for k in range(3, 13) if all(v % k == 0 for v in xs)]
        if ks:
            kmap[s] = max(ks); exs[s] = set(xs)
    d = df[df["number_set"].isin(list(kmap))]
    if len(d) == 0:
        return 0.0
    t = d["target"].astype(int).values
    sv = d["number_set"].values
    notex = np.array([tt not in exs[s] for s, tt in zip(sv, t)])
    k = d["number_set"].map(kmap).values
    mult = (t % k) == 0
    tmp = pd.DataFrame({"s": sv, "m": mult, "r": d["rating"].values})[notex]
    agg = tmp.groupby(["s", "m"])["r"].mean().unstack()
    if True not in agg.columns or False not in agg.columns:
        return 0.0
    diff = (agg[True] - agg[False]).dropna()
    return float(diff.mean()) if len(diff) else 0.0
