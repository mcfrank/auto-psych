# name: left_rate_matched_pairs
# description: Left-choice rate among pairs whose two sequences have identical switch rate and periodicity status (where the incumbent predicts exactly 0.5); observed above null_mean means people have a left-side response bias the model lacks.
def test_statistic(df):
    def sw(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y)
    def per(s):
        n = len(s)
        return int(any(all(s[i] == s[i + p] for i in range(n - p)) for p in range(1, n // 2 + 1)))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    swm = {s: sw(s) for s in u}
    pm_ = {s: per(s) for s in u}
    m = (df["sequence_a"].map(swm) == df["sequence_b"].map(swm)) & (df["sequence_a"].map(pm_) == df["sequence_b"].map(pm_))
    if m.sum() == 0:
        return 0.5
    return float(df["chose_left"][m].mean())
