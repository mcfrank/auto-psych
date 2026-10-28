# name: singleton_upward_minus_downward
# description: In one-example sets, yes-rate for targets 1-20 above the example minus targets 1-20 below it; observed above the null means people generalize upward from a lone number more than the model (which has no direction of travel and so is symmetric for singletons) predicts (below means a downward bias).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    single = ~ns.str.contains(",").values
    x = pd.to_numeric(ns.where(single, "0"), errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    up = single & (t > x) & (t - x <= 20)
    dn = single & (t < x) & (x - t <= 20)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(y[up].mean() - y[dn].mean())
