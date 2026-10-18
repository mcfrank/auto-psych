# name: modal_vs_minority_digit_match_gap
# description: In sets of >=3 examples where a final digit is shared by >=2 but not all examples, yes rate for targets ending in that modal digit minus yes rate for targets ending in a minority (single-example) digit (targets not examples); t_observed above null_mean means people adopt the majority "ends in d" rule and discount outlier examples more than the model's proportional digit averaging; below means less.
def test_statistic(df):
    info = {}
    for s in df["number_set"].astype(str).unique():
        xs = [int(x) for x in s.split(",")]
        ds = [x % 10 for x in xs]
        cnt = {d: ds.count(d) for d in ds}
        mx = max(cnt.values())
        ok = len(xs) >= 3 and mx >= 2 and mx < len(xs)
        modal = set(d for d, c in cnt.items() if c == mx)
        minor = set(d for d, c in cnt.items() if c == 1)
        info[s] = (ok, modal, minor, set(xs))
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    base = np.array([info[s][0] and (v not in info[s][3]) for s, v in zip(sets, t)])
    a = base & np.array([(v % 10) in info[s][1] for s, v in zip(sets, t)])
    b = base & np.array([(v % 10) in info[s][2] for s, v in zip(sets, t)])
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
