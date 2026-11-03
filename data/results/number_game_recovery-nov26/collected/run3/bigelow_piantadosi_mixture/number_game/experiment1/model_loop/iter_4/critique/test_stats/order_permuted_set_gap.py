# name: order_permuted_set_gap
# description: Mean absolute difference in yes rate, on shared targets, between sets containing the same numbers shown in a different order (e.g. "16,8" vs "8,16"); the model is order-invariant, so observed > null means presentation order changes people's generalisation beyond sampling noise.
def test_statistic(df):
    d = df[["number_set", "target", "rating"]].copy()
    sets = d["number_set"].unique()
    key = {s: ",".join(sorted(s.split(","), key=int)) for s in sets}
    d["key"] = d["number_set"].map(key)
    m = d.groupby(["key", "number_set", "target"])["rating"].mean().reset_index()
    n_orders = m.groupby(["key", "target"])["number_set"].transform("nunique")
    m = m[n_orders >= 2]
    if len(m) == 0:
        return 0.0
    rng = m.groupby(["key", "target"])["rating"].agg(lambda x: x.max() - x.min())
    return float(rng.mean())
