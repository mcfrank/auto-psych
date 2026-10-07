# name: participant_agreement_variance
# description: Variance across participants of each person's rate of agreeing with the pair's majority choice (pairs oriented canonically, majority from the data at hand); observed above null means people differ in consistency more than the person-lapse model produces, below means the lapse population over-disperses consistency.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    first = np.where(a <= b, a, b); second = np.where(a <= b, b, a)
    pick_first = np.where(a <= b, df["chose_left"].values, 1 - df["chose_left"].values)
    key = pd.Series(first) + "|" + pd.Series(second)
    rate = pd.Series(pick_first).groupby(key.values).transform("mean").values
    agree = np.where(rate >= 0.5, pick_first, 1 - pick_first)
    per = pd.Series(agree).groupby(df["participant_id"].values).mean()
    return float(per.var())
