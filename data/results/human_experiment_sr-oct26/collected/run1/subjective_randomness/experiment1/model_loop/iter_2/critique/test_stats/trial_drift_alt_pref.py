# name: trial_drift_alt_pref
# description: Proportion choosing the higher-alternation sequence (among pairs with different alternation rates) in the second half of each participant's trials minus the first half; nonzero observed vs a ~0 null means preferences drift over the session (learning, fatigue, or becoming noisier), which the stationary model does not predict.
def test_statistic(df):
    def alt(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    f = {s: alt(s) for s in u}
    d = df["sequence_a"].map(f).to_numpy(float) - df["sequence_b"].map(f).to_numpy(float)
    y = df["chose_left"].to_numpy(float)
    hi = np.where(d > 0, y, 1 - y)
    med = df.groupby("participant_id")["trial_index"].transform("median").to_numpy(float)
    late = df["trial_index"].to_numpy(float) > med
    m = np.abs(d) > 1e-9
    if (m & late).sum() == 0 or (m & ~late).sum() == 0:
        return 0.0
    return float(hi[m & late].mean() - hi[m & ~late].mean())
