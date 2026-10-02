# name: balance_vs_streak_tradeoff
# description: Among trials where one sequence is strictly more H/T-balanced but the other has a strictly shorter longest run, the proportion choosing the MORE BALANCED sequence; observed above the null means people weigh balance over streak length more than the model's imbalance and longest-run weights imply, below means they weigh streaks more.
def test_statistic(df):
    def feats(s):
        best = cur = 1
        for x, y in zip(s, s[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return abs(s.count("H") - s.count("T")) / len(s), best
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: feats(s) for s in u}
    ia = df["sequence_a"].map(lambda s: m[s][0]).to_numpy(); ib = df["sequence_b"].map(lambda s: m[s][0]).to_numpy()
    ra = df["sequence_a"].map(lambda s: m[s][1]).to_numpy(); rb = df["sequence_b"].map(lambda s: m[s][1]).to_numpy()
    a_bal = (ia < ib - 1e-9) & (ra > rb)
    b_bal = (ib < ia - 1e-9) & (rb > ra)
    sel = a_bal | b_bal
    if not sel.any():
        return 0.5
    cl = df["chose_left"].to_numpy()
    chose_bal = np.where(a_bal, cl, 1 - cl)
    return float(chose_bal[sel].mean())
