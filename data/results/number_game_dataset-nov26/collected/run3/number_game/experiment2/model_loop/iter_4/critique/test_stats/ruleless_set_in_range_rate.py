# name: ruleless_set_in_range_rate
# description: For 2+ example sets that no simple rule covers (no shared parity, last digit, contained digit, multiple of 3-12, square, cube or power of 2), yes rate on non-example targets inside the examples' range; observed above null means the model under-predicts interval-like generalization when no rule fits (below: over-predicts).
def test_statistic(df):
    rules = [set(range(1, 101, 2)), set(range(2, 101, 2)), set(k * k for k in range(1, 11)), {1, 8, 27, 64}, {1, 2, 4, 8, 16, 32, 64}]
    rules += [set(range(k, 101, k)) for k in range(3, 13)]
    rules += [set(v for v in range(1, 101) if v % 10 == d) for d in range(10)]
    rules += [set(v for v in range(1, 101) if str(d) in str(v)) for d in range(10)]
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        if len(ex) < 2:
            continue
        if not any(all(e in R for e in ex) for R in rules):
            info[s] = (min(ex), max(ex), set(ex))
    if not info:
        return 0.0
    keep = sets.isin(list(info.keys())).to_numpy()
    ss = sets[keep].to_numpy()
    t = df["target"].astype(int).to_numpy()[keep]
    r = df["rating"].to_numpy().astype(float)[keep]
    lo = np.array([info[a][0] for a in ss]); hi = np.array([info[a][1] for a in ss])
    isex = np.array([b in info[a][2] for a, b in zip(ss, t)], dtype=bool)
    m = (t >= lo) & (t <= hi) & ~isex
    return float(r[m].mean()) if m.sum() > 0 else 0.0
