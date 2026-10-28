# name: near_hull_upward_asymmetry
# description: For multi-example sets, yes rate on targets 1-15 above the largest example minus targets 1-15 below the smallest example; observed above null_mean means people extend generalization upward (to larger numbers) more than the direction-symmetric model predicts, below means they extend downward more.
def test_statistic(df):
    u = df["number_set"].unique()
    ex = {s: [int(x) for x in str(s).split(",")] for s in u}
    mx = df["number_set"].map({s: max(e) for s, e in ex.items()}).astype(int)
    mn = df["number_set"].map({s: min(e) for s, e in ex.items()}).astype(int)
    n = df["number_set"].map({s: len(e) for s, e in ex.items()}).astype(int)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    up = (n >= 2) & (t > mx) & (t - mx <= 15)
    dn = (n >= 2) & (t < mn) & (mn - t <= 15)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
