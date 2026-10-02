# name: participant_majority_agreement_sd
# description: SD across participants of each person's rate of agreeing with the pair-wise majority choice (majority computed from the same dataset); observed above the null means people differ in consistency/engagement more than the model's person-level weights and lapse rates produce, below means the model over-produces individual differences.
def test_statistic(df):
    a = df["sequence_a"].to_numpy()
    b = df["sequence_b"].to_numpy()
    first = np.where(a < b, a, b)
    second = np.where(a < b, b, a)
    c = df["chose_left"].to_numpy()
    chose_first = np.where(a < b, c, 1 - c)
    key = pd.Series(first) + "|" + pd.Series(second)
    pm_ = pd.Series(chose_first).groupby(key.to_numpy()).transform("mean").to_numpy()
    maj = (pm_ >= 0.5).astype(int)
    agree = (chose_first == maj).astype(float)
    r = pd.Series(agree).groupby(df["participant_id"].to_numpy()).mean()
    return float(r.std(ddof=0))
