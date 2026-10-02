# name: participant_majority_agreement_variance
# description: Variance across participants of each participant's rate of agreeing with the pair's overall majority choice (pair identified regardless of side); observed above null_mean means people differ in how consistently they follow the consensus more than the model's personal gain, sensitivity and lapse terms produce, below means the model now over-disperses individuals.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    sw = a < b
    key = np.char.add(np.char.add(np.where(sw, a, b), "|"), np.where(sw, b, a))
    cl = df["chose_left"].to_numpy().astype(float)
    cf = np.where(sw, cl, 1 - cl)
    pr = pd.Series(cf).groupby(key).transform("mean").to_numpy()
    agree = (cf == (pr >= 0.5)).astype(float)
    r = pd.Series(agree).groupby(df["participant_id"].to_numpy()).mean()
    return float(r.var(ddof=0))
