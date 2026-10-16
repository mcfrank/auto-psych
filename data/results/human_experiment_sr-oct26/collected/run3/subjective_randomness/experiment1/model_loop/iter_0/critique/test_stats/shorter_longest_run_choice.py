# name: shorter_longest_run_choice
# description: Among pairs whose longest runs differ, the rate of choosing the sequence with the shorter longest run; observed > null means people penalise long streaks more than the balance/alternation features predict (observed < null: less).
def test_statistic(df):
    def lr(s):
        best = cur = 1
        for i in range(1, len(s)):
            cur = cur + 1 if s[i] == s[i-1] else 1
            best = max(best, cur)
        return best
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    d = {s: lr(s) for s in u}
    ra = df["sequence_a"].map(d).values; rb = df["sequence_b"].map(d).values
    m = ra != rb
    if m.sum() == 0:
        return 0.5
    ch = np.where(ra < rb, df["chose_left"].values, 1 - df["chose_left"].values)
    return float(ch[m].mean())
