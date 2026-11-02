# name: above_vs_below_range
# description: Yes-rate for targets 1-20 above the largest example minus yes-rate for targets 1-20 below the smallest example; observed above null means people extend generalisation upward (toward larger numbers) more than the model, below null means they extend downward more.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    mx = {s: max(int(x) for x in s.split(",")) for s in uniq}
    mn = {s: min(int(x) for x in s.split(",")) for s in uniq}
    hi = sets.map(mx).to_numpy(); lo = sets.map(mn).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy()
    up = (t > hi) & (t - hi <= 20)
    dn = (t < lo) & (lo - t <= 20)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
