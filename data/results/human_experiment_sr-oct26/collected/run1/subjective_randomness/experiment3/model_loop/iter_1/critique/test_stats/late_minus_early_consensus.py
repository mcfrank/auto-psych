# name: late_minus_early_consensus
# description: Agreement with each pair's overall majority choice (pair identified regardless of side) in the second half of a session (trial_index > 32) minus the first half; observed above null_mean means responses become more decisive/consistent over the session (below: fatigue/noise grows), a drift the stationary model cannot produce.
def test_statistic(df):
    a = df["sequence_a"].to_numpy()
    b = df["sequence_b"].to_numpy()
    first = np.where(a < b, a, b)
    key = pd.Series(first + "|" + np.where(a < b, b, a))
    chose_first = np.where(a < b, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    pm_ = pd.Series(chose_first).groupby(key.to_numpy()).transform("mean").to_numpy()
    maj = (pm_ >= 0.5).astype(float)
    agree = (chose_first == maj).astype(float)
    late = df["trial_index"].to_numpy() > 32
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(agree[late].mean() - agree[~late].mean())
