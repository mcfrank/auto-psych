# name: single_example_fine_proximity
# description: For single-example sets and targets not sharing the example's final digit, yes-rate for targets within 10 of the example minus yes-rate for targets more than 40 away; observed > null means the model's coarse decade-band familiarity under-represents fine numerical-magnitude similarity.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    single = ~ns.str.contains(",")
    x = pd.to_numeric(ns.where(single, "0"), errors="coerce").fillna(0).astype(int)
    t = df["target"].astype(int)
    d = (t - x).abs()
    base = single & ((t % 10) != (x % 10))
    r = df["rating"].astype(float)
    a = r[base & (d <= 10)]
    b = r[base & (d > 40)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
