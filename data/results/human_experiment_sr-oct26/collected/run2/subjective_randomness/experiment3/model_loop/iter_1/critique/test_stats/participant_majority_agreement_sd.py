# name: participant_majority_agreement_sd
# description: SD across participants of each participant's rate of agreeing with the pair's majority choice; observed above null means people differ in consistency/noise more than the model allows (it under-produces heterogeneity in decision noise), below means less.
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    key = np.where(a < b, a + "|" + b, b + "|" + a)
    first = np.where(a < b, df["chose_left"], 1 - df["chose_left"])
    rate = pd.Series(first).groupby(key).transform("mean").to_numpy()
    agree = np.where(rate >= 0.5, first, 1 - first)
    per = pd.Series(agree).groupby(df["participant_id"].to_numpy()).mean()
    return float(per.std())
