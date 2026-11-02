# name: just_below_min_yes_rate
# description: Among sets of 2+ examples whose smallest example is >= 20, yes rate on targets 1-15 below the smallest example; observed above null means the model (ceiling above, but no graded floor below) under-produces downward extension near the minimum (below null = people impose a floor the model lacks).
def test_statistic(df):
    info = {}
    for s in df["number_set"].astype(str).unique():
        ex = [int(x) for x in s.split(",")]
        info[s] = (len(set(ex)), min(ex))
    ns = df["number_set"].astype(str).values
    k = np.array([info[s][0] for s in ns])
    mn = np.array([info[s][1] for s in ns])
    tg = df["target"].astype(int).values
    sel = (k >= 2) & (mn >= 20) & (tg < mn) & (tg >= mn - 15)
    if sel.sum() == 0:
        return 0.0
    return float(df["rating"].values[sel].mean())
