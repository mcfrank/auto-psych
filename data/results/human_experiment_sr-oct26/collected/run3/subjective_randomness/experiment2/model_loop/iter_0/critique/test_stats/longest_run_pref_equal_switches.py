# name: longest_run_pref_equal_switches
# description: Among pairs with equal switch counts but different longest runs, the rate of choosing the sequence with the shorter longest run as more random; observed above null_mean means people penalise long runs more than the model (which sees only switch counts, bias and periodicity).
def test_statistic(df):
    def lr(s):
        best = cur = 1
        for x, y in zip(s, s[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return best
    def sw(s):
        return sum(x != y for x, y in zip(s, s[1:]))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    LR = {s: lr(s) for s in u}; SW = {s: sw(s) for s in u}
    la = df["sequence_a"].map(LR).values; lb = df["sequence_b"].map(LR).values
    ka = df["sequence_a"].map(SW).values; kb = df["sequence_b"].map(SW).values
    m = (ka == kb) & (la != lb)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].values[m]
    a_short = (la < lb)[m]
    return float(np.mean(np.where(a_short, y, 1 - y)))
