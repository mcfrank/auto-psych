# name: low_agreement_participant_share
# description: Share of participants whose majority-agreement rate is below 0.6 (near-random responders); observed above null means the model under-produces near-random responders (lapse population too narrow), below means it over-produces them.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    first = np.where(a <= b, a, b); second = np.where(a <= b, b, a)
    pick_first = np.where(a <= b, df["chose_left"].values, 1 - df["chose_left"].values)
    key = pd.Series(first) + "|" + pd.Series(second)
    rate = pd.Series(pick_first).groupby(key.values).transform("mean").values
    agree = np.where(rate >= 0.5, pick_first, 1 - pick_first)
    per = pd.Series(agree).groupby(df["participant_id"].values).mean()
    return float((per < 0.6).mean())
