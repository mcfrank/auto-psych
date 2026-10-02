# name: shorter_longest_run_choice_rate
# description: Among trials whose two sequences differ in longest-run length by >=2, the fraction of choices of the sequence with the shorter longest run; observed > null means people penalise long streaks more than the model's balance/alternation features predict.
def test_statistic(df):
    def maxrun(s):
        best = cur = 1
        for i in range(1, len(s)):
            cur = cur + 1 if s[i] == s[i - 1] else 1
            best = max(best, cur)
        return best
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: maxrun(s) for s in u}
    ra = df["sequence_a"].map(m).to_numpy()
    rb = df["sequence_b"].map(m).to_numpy()
    y = df["chose_left"].to_numpy()
    mask = np.abs(ra - rb) >= 2
    if mask.sum() == 0:
        return 0.5
    chose_shorter = np.where(ra < rb, y, 1 - y)
    return float(chose_shorter[mask].mean())
