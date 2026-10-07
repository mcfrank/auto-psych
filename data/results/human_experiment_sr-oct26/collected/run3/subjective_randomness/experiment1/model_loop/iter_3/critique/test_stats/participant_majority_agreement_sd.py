# name: participant_majority_agreement_sd
# description: Standard deviation across participants of each participant's proportion of choices agreeing with the pair's majority choice (computed on the same dataset, ties counted as 0.5); observed above null_mean means individuals differ in which sequences they find random (or in consistency) more than the model's sensitivity-only heterogeneity produces, below means less.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    first = np.where(a < b, a, b); second = np.where(a < b, b, a)
    chose_first = np.where(a < b, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    key = pd.Series(first) + "|" + pd.Series(second)
    rate = pd.Series(chose_first).groupby(key).transform("mean").to_numpy()
    agree = np.where(rate > 0.5, chose_first, np.where(rate < 0.5, 1 - chose_first, 0.5))
    r = pd.Series(agree).groupby(df["participant_id"].to_numpy()).mean()
    return float(r.std(ddof=0))
