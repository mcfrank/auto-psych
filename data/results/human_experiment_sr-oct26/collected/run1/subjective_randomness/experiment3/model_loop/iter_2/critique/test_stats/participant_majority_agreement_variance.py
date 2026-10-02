# name: participant_majority_agreement_variance
# description: Variance across participants of each participant's rate of agreeing with the pair's overall majority choice (pair identified regardless of side); observed above null_mean means people differ in decisiveness/consistency more than the model's personal sensitivity and lapse terms produce, below means the model over-disperses individuals.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    first = np.where(a < b, a, b)
    key = pd.Series(first).str.cat(pd.Series(np.where(a < b, b, a)), sep="|")
    chose_first = np.where(a < b, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    pair_rate = pd.Series(chose_first).groupby(key.to_numpy()).transform("mean").to_numpy()
    maj = (pair_rate >= 0.5).astype(int)
    agree = (chose_first == maj).astype(float)
    return float(pd.Series(agree).groupby(df["participant_id"].to_numpy()).mean().var())
