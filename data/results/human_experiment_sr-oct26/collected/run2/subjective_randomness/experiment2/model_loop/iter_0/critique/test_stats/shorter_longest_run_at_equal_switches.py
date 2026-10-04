# name: shorter_longest_run_at_equal_switches
# description: Among pairs with equal switch counts but different longest-run lengths, the rate of choosing the sequence with the shorter longest run; observed above null means the model under-penalizes long streaks beyond what the switching rate implies.
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    def feat(s, f):
        return s.map({x: f(x) for x in s.unique()})
    def sw(x):
        return sum(1 for p, q in zip(x, x[1:]) if p != q)
    def lr(x):
        best = cur = 1
        for p, q in zip(x, x[1:]):
            cur = cur + 1 if p == q else 1
            best = max(best, cur)
        return best
    la, lb = feat(a, lr), feat(b, lr)
    mask = (feat(a, sw) == feat(b, sw)) & (la != lb)
    if mask.sum() == 0:
        return 0.5
    c = df["chose_left"][mask]
    return float(np.where((la < lb)[mask], c, 1 - c).mean())
