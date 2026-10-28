# name: single_example_above_minus_below
# description: For one-example sets, yes-rate on targets 1-15 above the example minus targets 1-15 below it; observed above null means people extrapolate upward from a single number more than the (symmetric-ish) interval prior predicts, below means they extrapolate downward more.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    single = (~ns.str.contains(",")).values
    e = pd.to_numeric(ns.where(~ns.str.contains(","), "0"), errors="coerce").fillna(0).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    d = t - e
    up = single & (d >= 1) & (d <= 15)
    dn = single & (d <= -1) & (d >= -15)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
