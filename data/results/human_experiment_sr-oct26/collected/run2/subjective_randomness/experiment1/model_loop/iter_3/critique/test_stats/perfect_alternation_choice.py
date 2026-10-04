# name: perfect_alternation_choice
# description: Among pairs where exactly one sequence perfectly alternates (HTHT.../THTH...), rate of choosing it; observed below null_mean means the model over-predicts preference for perfect alternation (people find it too regular), above means it over-penalises it.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    u = pd.unique(pd.concat([a, b]))
    pa_ = {s: float(all(x != y for x, y in zip(s, s[1:]))) for s in u}
    xa = a.map(pa_).to_numpy(); xb = b.map(pa_).to_numpy()
    m = xa != xb
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy(dtype=float)[m]
    return float(np.mean(np.where((xa > xb)[m], y, 1 - y)))
