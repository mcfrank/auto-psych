# name: participant_majority_agreement_sd
# description: Across-participant SD of each person's rate of agreeing with the item's majority choice; observed > null means people differ in how they judge randomness more than the single pooled model (one shared beta and weights) produces.
def test_statistic(df):
    a = df["sequence_a"].values.astype(str); b = df["sequence_b"].values.astype(str)
    flip = a > b
    first = np.where(flip, b, a); second = np.where(flip, a, b)
    chose_first = np.where(flip, 1 - df["chose_left"].values, df["chose_left"].values)
    key = pd.Series(first) + "|" + pd.Series(second)
    m = pd.Series(chose_first).groupby(key.values).transform("mean").values
    agree = np.where(m >= 0.5, chose_first, 1 - chose_first)
    return float(pd.Series(agree).groupby(df["participant_id"].values).mean().std())
