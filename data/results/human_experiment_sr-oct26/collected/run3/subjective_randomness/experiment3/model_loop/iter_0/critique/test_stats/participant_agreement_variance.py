# name: participant_agreement_variance
# description: Variance across participants of each person's rate of agreeing with the pair majority choice (heterogeneity of individual judgment sharpness); observed above null means people differ in consistency more than the model's person-level sensitivity produces (e.g. a subgroup of random responders), below means less.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str)
    b = df["sequence_b"].to_numpy().astype(str)
    a_first = a <= b
    y = df["chose_left"].to_numpy()
    chose_x = np.where(a_first, y, 1 - y)
    key = pd.Series(np.where(a_first, a, b)) + "|" + pd.Series(np.where(a_first, b, a))
    rate = pd.Series(chose_x).groupby(key.to_numpy()).transform("mean").to_numpy()
    agree = (chose_x == (rate >= 0.5)).astype(float)
    per = pd.Series(agree).groupby(df["participant_id"].to_numpy()).mean()
    return float(per.var())
