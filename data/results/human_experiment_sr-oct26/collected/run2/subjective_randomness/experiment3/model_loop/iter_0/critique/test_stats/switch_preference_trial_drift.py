# name: switch_preference_trial_drift
# description: Difference (second half minus first half of each session by trial_index) in the rate of choosing the more-switching sequence among unequal-switch pairs; observed away from null (~0) means the switching preference drifts over the session, which the model's static weights cannot produce.
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    sw = {s: sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1) for s in seqs}
    ra = df["sequence_a"].map(sw).to_numpy(); rb = df["sequence_b"].map(sw).to_numpy()
    m = ra != rb
    y = np.where(ra > rb, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    t = df["trial_index"].to_numpy()
    med = df.groupby("participant_id")["trial_index"].transform("median").to_numpy()
    late = t > med
    a = y[m & late]; b = y[m & ~late]
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(a.mean() - b.mean())
