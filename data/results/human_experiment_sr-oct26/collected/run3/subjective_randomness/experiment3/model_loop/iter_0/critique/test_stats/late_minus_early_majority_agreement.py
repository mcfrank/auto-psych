# name: late_minus_early_majority_agreement
# description: Agreement with the pair's overall majority choice in the second half of each session minus in the first half; observed below null means judgments become noisier (fatigue) over the session more than the model's constant sensitivity predicts, above means they sharpen.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str)
    b = df["sequence_b"].to_numpy().astype(str)
    a_first = a <= b
    chose_x = np.where(a_first, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    key = pd.Series(np.where(a_first, a, b)) + "|" + pd.Series(np.where(a_first, b, a))
    rate = pd.Series(chose_x).groupby(key.to_numpy()).transform("mean").to_numpy()
    maj = (rate >= 0.5).astype(int)
    agree = (chose_x == maj).astype(float)
    ti = df["trial_index"].to_numpy()
    med = pd.Series(ti).groupby(df["participant_id"].to_numpy()).transform("median").to_numpy()
    late = ti > med
    return float(agree[late].mean() - agree[~late].mean())
