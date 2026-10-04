# name: shorter_longest_run_pref
# description: Among pairs with equal switch counts but different longest-run lengths, rate of choosing the sequence with the shorter longest run; observed above null_mean means the model under-penalises long streaks beyond what switch rate and periodicity explain.
def test_statistic(df):
    def sw(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y)
    def lr(s):
        best = cur = 1
        for x, y in zip(s, s[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return best
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    swm = {s: sw(s) for s in u}
    lrm = {s: lr(s) for s in u}
    sa, sb = df["sequence_a"].map(swm), df["sequence_b"].map(swm)
    la, lb = df["sequence_a"].map(lrm), df["sequence_b"].map(lrm)
    m = (sa == sb) & (la != lb)
    if m.sum() == 0:
        return 0.5
    chose_short = np.where(la[m] < lb[m], df["chose_left"][m], 1 - df["chose_left"][m])
    return float(np.mean(chose_short))
