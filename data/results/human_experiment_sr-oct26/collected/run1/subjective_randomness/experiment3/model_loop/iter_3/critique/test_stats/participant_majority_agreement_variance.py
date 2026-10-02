# name: participant_majority_agreement_variance
# description: Variance across participants of each participant's rate of agreeing with the pair's overall majority choice (pair identified regardless of side); observed above null_mean means people still differ in how consistently they follow the consensus more than the model's personal terms produce.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    sw = a < b
    key = np.char.add(np.char.add(np.where(sw, a, b), "|"), np.where(sw, b, a))
    cl = df["chose_left"].to_numpy().astype(float)
    chose_first = np.where(sw, cl, 1 - cl)
    pair_rate = pd.Series(chose_first).groupby(key).transform("mean").to_numpy()
    agree = (chose_first == (pair_rate >= 0.5)).astype(float)
    return float(pd.Series(agree).groupby(df["participant_id"].to_numpy()).mean().var())
