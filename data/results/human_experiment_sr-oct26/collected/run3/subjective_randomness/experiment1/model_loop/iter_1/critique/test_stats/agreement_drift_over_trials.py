# name: agreement_drift_over_trials
# description: Agreement with each pair's data-majority choice in the second half of each participant's session minus the first half; observed below null_mean means people grow noisier/less consistent over the session (fatigue), above means they sharpen — the model assumes a constant sensitivity over trials.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    first_is_a = a < b
    key = np.where(first_is_a, a, b) + "|" + np.where(first_is_a, b, a)
    y = df["chose_left"].values
    cf = np.where(first_is_a, y, 1 - y)
    pm_ = pd.Series(cf).groupby(key).transform("mean").values
    agree = (cf == (pm_ >= 0.5)).astype(float)
    t = df["trial_index"].values
    med = df.groupby("participant_id")["trial_index"].transform("median").values
    late = t > med
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(agree[late].mean() - agree[~late].mean())
