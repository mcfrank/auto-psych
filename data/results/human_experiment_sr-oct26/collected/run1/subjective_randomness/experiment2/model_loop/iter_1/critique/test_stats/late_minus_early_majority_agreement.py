# name: late_minus_early_majority_agreement
# description: Agreement with each pair's majority choice (over all participants, side-aligned) in the second half of a participant's trials minus the first half; negative observed vs null means responses get noisier with time (fatigue) beyond the model's constant lapse rate, positive means they sharpen.
def test_statistic(df):
    a = df["sequence_a"].to_numpy()
    b = df["sequence_b"].to_numpy()
    first = np.where(a < b, a, b)
    key = pd.Series(first + "|" + np.where(a < b, b, a))
    c = df["chose_left"].to_numpy()
    chose_first = np.where(a < b, c, 1 - c)
    pm_ = pd.Series(chose_first).groupby(key.to_numpy()).transform("mean").to_numpy()
    maj = (pm_ > 0.5).astype(int)
    tie = pm_ == 0.5
    agree = (chose_first == maj).astype(float)
    rank = df.groupby("participant_id")["trial_index"].rank(pct=True).to_numpy()
    late = (rank > 0.5) & ~tie
    early = (rank <= 0.5) & ~tie
    if late.sum() == 0 or early.sum() == 0:
        return 0.0
    return float(agree[late].mean() - agree[early].mean())
