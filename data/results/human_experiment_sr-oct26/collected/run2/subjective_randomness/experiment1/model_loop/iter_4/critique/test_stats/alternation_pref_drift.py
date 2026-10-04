# name: alternation_pref_drift
# description: Rate of choosing the higher-switch-rate sequence in each participant's second half of trials minus their first half (pairs with different rates); the model is stationary (null near 0), so observed below/above null_mean means preference for alternation weakens/strengthens over the session.
def test_statistic(df):
    def alt(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    am = {s: alt(s) for s in u}
    a, b = df["sequence_a"].map(am).values, df["sequence_b"].map(am).values
    c = df["chose_left"].values
    chose_hi = np.where(a > b, c, 1 - c)
    med = df.groupby("participant_id")["trial_index"].transform("median").values
    late = df["trial_index"].values > med
    m = a != b
    r2 = chose_hi[m & late].mean() if (m & late).sum() else 0.5
    r1 = chose_hi[m & ~late].mean() if (m & ~late).sum() else 0.5
    return float(r2 - r1)
