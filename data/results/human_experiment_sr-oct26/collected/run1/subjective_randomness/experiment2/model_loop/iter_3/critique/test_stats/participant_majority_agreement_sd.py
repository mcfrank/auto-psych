# name: participant_majority_agreement_sd
# description: SD across participants of each person's rate of agreeing with the pair-wise majority choice (majority from the same dataset, tied pairs skipped); observed above the null means people still differ in consistency more than the model's person-level sensitivity and lapse rates produce, below means the model over-produces individual differences.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    first = np.where(a < b, a, b)
    key = pd.Series(np.char.add(np.char.add(first, "|"), np.where(a < b, b, a)))
    chose_first = (np.where(df["chose_left"].to_numpy() == 1, a, b) == first).astype(float)
    rate = pd.Series(chose_first).groupby(key).transform("mean").to_numpy()
    ok = rate != 0.5
    agree = np.where(rate > 0.5, chose_first, 1 - chose_first)
    s = pd.Series(agree[ok]).groupby(df["participant_id"].to_numpy()[ok]).mean()
    return float(s.std())
