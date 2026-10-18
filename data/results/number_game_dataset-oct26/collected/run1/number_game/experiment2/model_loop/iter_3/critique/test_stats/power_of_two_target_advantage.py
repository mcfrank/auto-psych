# name: power_of_two_target_advantage
# description: In sets of >=2 examples that are all powers of 2, yes rate for power-of-2 targets minus yes rate for other even targets (targets not examples); t_observed above null_mean means people infer a "powers of 2" rule the model (final digit + common-factor multiples) under-produces; below means the model over-favours powers of 2.
def test_statistic(df):
    p2 = {1, 2, 4, 8, 16, 32, 64}
    info = {}
    for s in df["number_set"].astype(str).unique():
        xs = [int(x) for x in s.split(",")]
        info[s] = (len(xs) >= 2 and all(x in p2 for x in xs), set(xs))
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    ok = np.array([info[s][0] and (v not in info[s][1]) for s, v in zip(sets, t)])
    isp2 = np.isin(t, list(p2))
    even = (t % 2 == 0)
    r = df["rating"].astype(float).values
    a = ok & isp2
    b = ok & even & ~isp2
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
