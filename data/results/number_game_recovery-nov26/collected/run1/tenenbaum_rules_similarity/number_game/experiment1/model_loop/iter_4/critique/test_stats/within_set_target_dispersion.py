# name: within_set_target_dispersion
# description: For sets with >= 100 responses, the mean over sets of the standard deviation across targets (non-example, >=5 responses each) of the per-target yes-rate; observed > null means people discriminate among targets within a heavily-probed set (e.g. 52,22,94 -> 16, 44 always yes, 98, 6 mostly no) more than the model, which gives nearly flat predictions across same-parity targets.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    cnt = ns.map(ns.value_counts())
    t = df["target"].astype(int)
    sets = ns.str.split(",").apply(lambda l: [int(x) for x in l])
    inset = pd.Series([x in l for x, l in zip(t, sets)], index=df.index)
    sub = df[(cnt >= 100) & ~inset]
    if sub.empty:
        return 0.0
    g = sub.groupby([sub["number_set"].astype(str), sub["target"].astype(int)])["rating"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= 5]
    sd = g.groupby("number_set")["mean"].std().dropna()
    return float(sd.mean()) if len(sd) else 0.0
