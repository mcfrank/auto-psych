# name: longer_streak_at_equal_alternation
# description: Among trials where both sequences have the same number of alternations but different longest-run lengths, the proportion choosing the sequence with the LONGER longest run; observed below the null means people penalise a concentrated streak (vs. spread-out repeats) more than the model's longest-run weight predicts, above means less.
def test_statistic(df):
    def stats(s):
        alt = sum(1 for x, y in zip(s, s[1:]) if x != y)
        best = cur = 1
        for x, y in zip(s, s[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return alt, best
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    ma = {s: stats(s)[0] for s in u}
    mr = {s: stats(s)[1] for s in u}
    aa = df["sequence_a"].map(ma).to_numpy()
    ab = df["sequence_b"].map(ma).to_numpy()
    ra = df["sequence_a"].map(mr).to_numpy()
    rb = df["sequence_b"].map(mr).to_numpy()
    sel = (aa == ab) & (ra != rb)
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(ra > rb, c, 1 - c)
    return float(chose[sel].mean())
