# name: participant_majority_agreement_sd
# description: Standard deviation across participants of each participant's proportion of choices agreeing with the pair's majority choice (computed on the same dataset, ties as 0.5); observed above null_mean means individuals differ in which sequences they find random more than the model's sensitivity-only heterogeneity produces, below means less.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    lo = np.where(a < b, a, b); key = pd.Series(lo + "|" + np.where(a < b, b, a), index=df.index)
    chose_lo = np.where(a.to_numpy() == lo, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy()).astype(float)
    s = pd.Series(chose_lo, index=df.index)
    rate = s.groupby(key).transform("mean")
    agree = np.where(rate > 0.5, chose_lo, np.where(rate < 0.5, 1 - chose_lo, 0.5))
    return float(pd.Series(agree, index=df.index).groupby(df["participant_id"]).mean().std(ddof=0))
