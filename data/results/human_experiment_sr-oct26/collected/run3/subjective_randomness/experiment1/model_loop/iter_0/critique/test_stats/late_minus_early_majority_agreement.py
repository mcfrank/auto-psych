# name: late_minus_early_majority_agreement
# description: Agreement with the item majority choice in the second half of each session (trial_index above the participant median) minus the first half; negative-vs-null means responses get noisier with fatigue, which the model's stationary beta does not produce.
def test_statistic(df):
    a = df["sequence_a"].values.astype(str); b = df["sequence_b"].values.astype(str)
    flip = a > b
    first = np.where(flip, b, a); second = np.where(flip, a, b)
    chose_first = np.where(flip, 1 - df["chose_left"].values, df["chose_left"].values)
    key = pd.Series(first) + "|" + pd.Series(second)
    m = pd.Series(chose_first).groupby(key.values).transform("mean").values
    agree = np.where(m >= 0.5, chose_first, 1 - chose_first)
    t = df["trial_index"].values
    med = df.groupby("participant_id")["trial_index"].transform("median").values
    late = t > med
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(agree[late].mean() - agree[~late].mean())
