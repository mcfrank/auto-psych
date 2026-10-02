# name: late_minus_early_alternation_preference
# description: Rate of choosing the more alternating sequence (trials where alternation rates differ) in each participant's second half of trials minus the first half; positive observed above the null means preferences for alternation strengthen over the session (learning/sharpening) beyond the model's time-invariant judgement, below means they fade (fatigue/increasing guessing).
def test_statistic(df):
    def alt(s):
        return sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: alt(s) for s in u}
    aa = df["sequence_a"].map(m).to_numpy(); ab = df["sequence_b"].map(m).to_numpy()
    cl = df["chose_left"].to_numpy()
    med = df.groupby("participant_id")["trial_index"].transform("median").to_numpy()
    late = df["trial_index"].to_numpy() > med
    sel = np.abs(aa - ab) > 1e-9
    chose_more = np.where(aa > ab, cl, 1 - cl)
    e = sel & ~late; l = sel & late
    if not e.any() or not l.any():
        return 0.0
    return float(chose_more[l].mean() - chose_more[e].mean())
