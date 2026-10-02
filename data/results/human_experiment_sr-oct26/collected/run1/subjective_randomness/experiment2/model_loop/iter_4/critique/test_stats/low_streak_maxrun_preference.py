# name: low_streak_maxrun_preference
# description: Among trials where both sequences have a longest run <= 3 and the longest runs differ, the proportion choosing the sequence with the SHORTER longest run; observed below the null means people do not penalise short streaks (2-3) as much as the model's linear longest-run term implies (a threshold-like streak aversion), above means they penalise them more.
def test_statistic(df):
    def mr(s):
        best = cur = 1
        for x, y in zip(s, s[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return best
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: mr(s) for s in u}
    ra = df["sequence_a"].map(m).to_numpy(); rb = df["sequence_b"].map(m).to_numpy()
    sel = (ra <= 3) & (rb <= 3) & (ra != rb)
    if not sel.any():
        return 0.5
    cl = df["chose_left"].to_numpy()
    chose_short = np.where(ra < rb, cl, 1 - cl)
    return float(chose_short[sel].mean())
