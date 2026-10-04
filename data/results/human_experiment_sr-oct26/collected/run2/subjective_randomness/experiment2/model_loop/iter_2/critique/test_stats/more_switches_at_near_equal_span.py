# name: more_switches_at_near_equal_span
# description: Among pairs whose tally spans differ by at most 1, the rate of choosing the sequence with more H/T switches; observed above null means the model under-predicts a preference for alternation.

def _feat(s):
    t = 0; hi = 0; lo = 0; run = 1; best = 1; sw = 0
    for i, c in enumerate(s):
        t += 1 if c == "H" else -1
        hi = max(hi, t); lo = min(lo, t)
        if i > 0:
            if c == s[i - 1]:
                run += 1
            else:
                run = 1; sw += 1
            best = max(best, run)
    tri = len(set(s[i:i + 3] for i in range(len(s) - 2)))
    return {"span": hi - lo, "run": best, "sw": sw, "nh": s.count("H"), "imb": abs(2 * s.count("H") - len(s)), "tri": tri}

def _f(df, key):
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: _feat(s)[key] for s in u}
    return df["sequence_a"].map(m).to_numpy(float), df["sequence_b"].map(m).to_numpy(float)

def _pick_rate(df, fa, fb, mask, a_is_target):
    y = df["chose_left"].to_numpy(float)
    mask = mask & (fa != fb)
    if mask.sum() == 0:
        return 0.5
    chose_t = np.where(a_is_target, y, 1 - y)
    return float(chose_t[mask].mean())


def test_statistic(df):
    sa, sb = _f(df, "span"); wa, wb = _f(df, "sw")
    return _pick_rate(df, wa, wb, np.abs(sa - sb) <= 1, wa > wb)
