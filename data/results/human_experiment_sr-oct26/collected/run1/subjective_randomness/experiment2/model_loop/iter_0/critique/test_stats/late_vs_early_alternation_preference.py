# name: late_vs_early_alternation_preference
# description: Proportion choosing the sequence with more alternations in the second half of each participant's trials minus that in the first half (pairs with unequal alternation counts); observed != null means the preference drifts over the session (fatigue/learning) while the model's weights are fixed across trials.
def test_statistic(df):
    def nalt(s):
        s = str(s)
        return sum(1 for x, y in zip(s, s[1:]) if x != y)
    ua = df["sequence_a"].map({u: nalt(u) for u in df["sequence_a"].unique()}).values
    ub = df["sequence_b"].map({u: nalt(u) for u in df["sequence_b"].unique()}).values
    rank = df.groupby("participant_id")["trial_index"].rank(pct=True).values
    m = ua != ub
    c = df["chose_left"].values
    chose_more = np.where(ua > ub, c, 1 - c)
    late = m & (rank > 0.5)
    early = m & (rank <= 0.5)
    if late.sum() == 0 or early.sum() == 0:
        return 0.0
    return float(chose_more[late].mean() - chose_more[early].mean())
