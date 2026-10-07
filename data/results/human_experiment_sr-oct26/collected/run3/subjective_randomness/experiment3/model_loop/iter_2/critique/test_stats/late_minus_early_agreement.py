# name: late_minus_early_agreement
# description: Agreement with the pair's majority choice on trials in the second half of each session (trial_index > 32) minus the first half; observed below null (null ~0) means judgments become noisier/less consensual over the session (fatigue), which the time-invariant model cannot produce; above means they sharpen with practice.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    first = np.where(a <= b, a, b); second = np.where(a <= b, b, a)
    pick_first = np.where(a <= b, df["chose_left"].values, 1 - df["chose_left"].values)
    key = pd.Series(first) + "|" + pd.Series(second)
    rate = pd.Series(pick_first).groupby(key.values).transform("mean").values
    agree = np.where(rate >= 0.5, pick_first, 1 - pick_first).astype(float)
    late = df["trial_index"].values > 32
    if late.all() or (~late).all():
        return 0.0
    return float(agree[late].mean() - agree[~late].mean())
