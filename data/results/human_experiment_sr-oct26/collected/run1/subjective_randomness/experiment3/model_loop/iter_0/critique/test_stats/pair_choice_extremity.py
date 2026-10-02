# name: pair_choice_extremity
# description: Mean over distinct stimulus pairs of |pair-level choice proportion - 0.5|; observed above null_mean means people agree on pairs more strongly than the model predicts (model too noisy / under-confident), below means the model is over-confident.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    canon_left = a <= b
    key = np.where(canon_left, a, b) + "|" + np.where(canon_left, b, a)
    c = df["chose_left"].to_numpy()
    chose_canon = np.where(canon_left, c, 1 - c)
    rate = pd.Series(chose_canon).groupby(key).mean()
    return float((rate - 0.5).abs().mean())
