# name: participant_majority_agreement_sd
# description: Across-participant SD of each person's rate of agreeing with the per-pair majority choice (computed on the same dataset); observed above null_mean means people differ in consistency/strategy more than the model's ideal-switch-rate heterogeneity produces, below means the model over-produces individual differences.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    y = df["chose_left"].to_numpy(dtype=float)
    swap = a > b
    key = np.where(swap, b + "|" + a, a + "|" + b)
    pick_first = np.where(swap, 1 - y, y)
    s = pd.Series(pick_first)
    share = s.groupby(key).transform("mean").to_numpy()
    agree = np.where(share >= 0.5, pick_first, 1 - pick_first)
    return float(pd.Series(agree).groupby(df["participant_id"].to_numpy()).mean().std(ddof=0))
