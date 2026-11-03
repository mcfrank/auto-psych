# name: decade_crossing_near_contrast
# description: For non-example targets 1-5 from the nearest example, yes rate when the target is in the same tens decade (floor((n-1)/10)) as that example minus when it crosses a decade boundary; observed above the null means people treat decades as category boundaries the model ignores (below = decade boundaries matter less than the model predicts).
def _rules():
    r = {"squares": set(n * n for n in range(1, 11)), "cubes": set(n ** 3 for n in range(1, 5)),
         "pow2": {1, 2, 4, 8, 16, 32, 64}}
    for m in range(3, 13):
        r["m%d" % m] = set(range(m, 101, m))
    for d in range(10):
        r["e%d" % d] = set(n for n in range(1, 101) if n % 10 == d)
    return r


def _set_info(df):
    # Per unique set: examples, smallest familiar rule (size <= 16) containing all examples.
    rules = _rules()
    info = {}
    for s in df["number_set"].unique():
        ex = [int(x) for x in str(s).split(",")]
        best = None
        for k, mem in rules.items():
            if all(e in mem for e in ex) and (best is None or len(mem) < len(best)):
                best = mem
        info[s] = (ex, best)
    return info


def _cols(df):
    info = _set_info(df)
    sets = df["number_set"].values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    n_ex = np.array([len(info[s][0]) for s in sets])
    lo = np.array([min(info[s][0]) for s in sets])
    hi = np.array([max(info[s][0]) for s in sets])
    in_set = np.array([tt in info[s][0] for s, tt in zip(sets, t)])
    has_rule = np.array([info[s][1] is not None for s in sets])
    member = np.array([info[s][1] is not None and tt in info[s][1] for s, tt in zip(sets, t)])
    exarr = np.full((len(t), 4), np.nan)
    for i, s in enumerate(sets):
        exarr[i, :len(info[s][0])] = info[s][0]
    dist = np.nanmin(np.abs(exarr - t[:, None]), axis=1)
    near_ex = exarr[np.arange(len(t)), np.nanargmin(np.abs(exarr - t[:, None]), axis=1)]
    return dict(t=t, y=y, n_ex=n_ex, lo=lo, hi=hi, in_set=in_set, has_rule=has_rule,
                member=member, dist=dist, near_ex=near_ex)


def _diff(y, a, b):
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())

def test_statistic(df):
    c = _cols(df)
    base = ~c["in_set"] & (c["dist"] >= 1) & (c["dist"] <= 5)
    same = (c["t"] - 1) // 10 == (c["near_ex"].astype(int) - 1) // 10
    return _diff(c["y"], base & same, base & ~same)
