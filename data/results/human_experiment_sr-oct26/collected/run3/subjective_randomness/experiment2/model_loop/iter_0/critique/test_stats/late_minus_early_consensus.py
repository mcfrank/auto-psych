# name: late_minus_early_consensus
# description: Agreement with each pair's majority choice in the second half of each participant's trials minus the first half; positive observed vs null_mean means people become more consistent/decisive over the session (negative: fatigue), which the stationary model cannot produce.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    first = np.where(a < b, a, b)
    key = pd.Series(first) + "|" + pd.Series(np.where(a < b, b, a))
    chose_first = np.where(a < b, df["chose_left"].values, 1 - df["chose_left"].values)
    rate = pd.Series(chose_first).groupby(key.values).transform("mean").values
    maj = (rate >= 0.5).astype(int)
    agree = (chose_first == maj).astype(float)
    med = df.groupby("participant_id")["trial_index"].transform("median").values
    late = df["trial_index"].values > med
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(agree[late].mean() - agree[~late].mean())
