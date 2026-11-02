# name: rule_set_nonmember_inside_span_yes
# description: In sets of 2+ examples sharing a small familiar rule, yes rate for targets inside the examples' span that are NOT rule members; observed above the null means people mix in magnitude-range generalisation alongside the rule more than the model does (below = they commit to the rule more strictly than the model).
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
    m = (c["n_ex"] >= 2) & c["has_rule"] & ~c["member"] & ~c["in_set"] & (c["t"] > c["lo"]) & (c["t"] < c["hi"])
    return float(c["y"][m].mean()) if m.sum() else 0.0
