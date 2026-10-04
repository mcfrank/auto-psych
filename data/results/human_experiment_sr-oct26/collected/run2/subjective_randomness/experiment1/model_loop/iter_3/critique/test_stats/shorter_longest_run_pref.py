# name: shorter_longest_run_pref
# description: Among pairs with equal switch counts but different longest-run lengths, rate of choosing the sequence with the shorter longest run; observed above null_mean means the model under-penalises long streaks beyond what switch rate (and periodicity) explain.
def test_statistic(df):
    def lr(s):
        best = cur = 1
        for i in range(1, len(s)):
            cur = cur + 1 if s[i] == s[i - 1] else 1
            best = max(best, cur)
        return best
    a = df["sequence_a"]; b = df["sequence_b"]
    u = pd.unique(pd.concat([a, b]))
    d = {s: lr(s) for s in u}
    sw = {s: sum(1 for x, y in zip(s, s[1:]) if x != y) for s in u}
    ra = a.map(d).to_numpy(); rb = b.map(d).to_numpy()
    sa = a.map(sw).to_numpy(); sb = b.map(sw).to_numpy()
    m = (sa == sb) & (ra != rb)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy(dtype=float)[m]
    return float(np.mean(np.where((ra < rb)[m], y, 1 - y)))
