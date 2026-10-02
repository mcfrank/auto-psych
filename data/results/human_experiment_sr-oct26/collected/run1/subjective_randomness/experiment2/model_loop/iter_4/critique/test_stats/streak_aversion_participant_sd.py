# name: streak_aversion_participant_sd
# description: SD across participants of each person's rate of choosing the sequence with the shorter longest run (over trials where the longest runs differ); observed above the null means people differ in streak aversion more than the model's personal longest-run weights and lapse rates produce, below means the model over-produces this individual variation.
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
    sel = ra != rb
    if not sel.any():
        return 0.0
    cl = df["chose_left"].to_numpy()
    chose_short = np.where(ra < rb, cl, 1 - cl)
    rates = pd.Series(chose_short[sel]).groupby(df["participant_id"].to_numpy()[sel]).mean()
    if len(rates) < 2:
        return 0.0
    return float(rates.std())
