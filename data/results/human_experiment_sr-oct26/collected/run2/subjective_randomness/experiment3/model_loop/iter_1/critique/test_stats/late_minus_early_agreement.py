# name: late_minus_early_agreement
# description: Rate of agreeing with the pair's majority choice in the second half of each session (trial_index above the participant's median) minus the first half; observed below null means responses get noisier with fatigue, which the time-invariant model does not produce; above means they sharpen.
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    key = np.where(a < b, a + "|" + b, b + "|" + a)
    first = np.where(a < b, df["chose_left"], 1 - df["chose_left"])
    rate = pd.Series(first).groupby(key).transform("mean").to_numpy()
    agree = np.where(rate >= 0.5, first, 1 - first)
    t = df["trial_index"].to_numpy()
    med = df.groupby("participant_id")["trial_index"].transform("median").to_numpy()
    late = t > med
    if late.all() or (~late).all():
        return 0.0
    return float(agree[late].mean() - agree[~late].mean())
