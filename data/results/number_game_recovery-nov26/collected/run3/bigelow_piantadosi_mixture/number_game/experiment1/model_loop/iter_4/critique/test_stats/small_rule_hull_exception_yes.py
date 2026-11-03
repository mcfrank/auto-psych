# name: small_rule_hull_exception_yes
# description: Among sets of 2+ examples that all fall in one small rule (<= 12 members: squares, cubes, powers of k, multiples of 9/11/12), yes rate for non-rule targets lying inside the examples' range; observed > null means people fill in the numeric range around tight-rule sets more than the model (its size principle is too sharp), observed < null means less.
def test_statistic(df):
    rules = [[n * n for n in range(1, 11)], [n ** 3 for n in range(1, 5)]]
    rules += [[m ** k for k in range(1, 8) if m ** k <= 100] for m in range(2, 11)]
    rules += [list(range(m, 101, m)) for m in [9, 11, 12]]
    rules = [set(r) for r in rules if len(r) >= 2]
    sets = df["number_set"].unique()
    info = {}
    for s in sets:
        l = [int(x) for x in s.split(",")]
        if len(l) < 2:
            continue
        fits = [r for r in rules if set(l) <= r]
        if fits:
            info[s] = (min(l), max(l), set().union(*fits))
    if not info:
        return 0.0
    sub = df[df["number_set"].isin(list(info))]
    items = sub.groupby(["number_set", "target"])["rating"].mean().reset_index()
    m = np.array([(info[s][0] < t < info[s][1]) and (t not in info[s][2])
                  for s, t in zip(items["number_set"], items["target"].astype(int))])
    if m.sum() == 0:
        return 0.0
    return float(items["rating"][m].mean())
