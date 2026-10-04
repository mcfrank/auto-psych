# name: late_minus_early_majority_agreement
# description: Agreement with each pair's population-majority choice in the second half of a participant's trials minus the first half; positive vs null means responses grow more consistent over the session (negative: fatigue/noise grows), a drift the stationary model cannot produce.
def test_statistic(df):
    key = df["sequence_a"] + "|" + df["sequence_b"]
    pm_ = df.groupby(key)["chose_left"].transform("mean").to_numpy()
    c = df["chose_left"].to_numpy()
    agree = np.where(pm_ >= 0.5, c, 1 - c)
    med = df.groupby("participant_id")["trial_index"].transform("median").to_numpy()
    late = df["trial_index"].to_numpy() > med
    return float(agree[late].mean() - agree[~late].mean())
