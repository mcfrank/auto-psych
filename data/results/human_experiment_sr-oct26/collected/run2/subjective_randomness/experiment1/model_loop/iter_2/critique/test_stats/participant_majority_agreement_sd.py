# name: participant_majority_agreement_sd
# description: SD across participants of their agreement rate with the pair-level majority choice; observed above null_mean means the model under-produces individual differences in consistency with the group.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(dtype=object); b = df["sequence_b"].to_numpy(dtype=object)
    y = df["chose_left"].to_numpy(dtype=float)
    first_left = a <= b
    key = np.where(first_left, a + "|" + b, b + "|" + a)
    c = np.where(first_left, y, 1 - y)
    pm_ = pd.Series(c).groupby(key).transform("mean").to_numpy()
    agree = np.where(pm_ >= 0.5, c, 1 - c)
    r = pd.Series(agree).groupby(df["participant_id"].to_numpy()).mean()
    return float(np.std(r.to_numpy()))
