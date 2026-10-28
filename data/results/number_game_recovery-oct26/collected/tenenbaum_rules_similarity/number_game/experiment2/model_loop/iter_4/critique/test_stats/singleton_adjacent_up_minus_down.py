# name: singleton_adjacent_up_minus_down
# description: On single-example sets, yes-rate to targets 1-3 above the example minus yes-rate to targets 1-3 below it; observed above null means the top-marking model's downward-only generalization from one example is too strong right next to the example (people extend upward more), below null means it is too weak.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    single = ~sets.str.contains(",")
    x = pd.to_numeric(sets.where(single, "0"), errors="coerce").fillna(0).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    d = t - x
    single = single.to_numpy()
    up = single & (d >= 1) & (d <= 3)
    dn = single & (d <= -1) & (d >= -3)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
