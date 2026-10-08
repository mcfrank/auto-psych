# name: pair_choice_extremity
# description: Mean over unordered stimulus pairs of |P(choose the lexicographically-first sequence) - 0.5|, pooling both presentation sides; observed above null_mean means the population consensus per pair is stronger than the model predicts (model too noisy/flat), below means the model is overconfident.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    first_is_a = a < b
    key = np.where(first_is_a, a, b) + "|" + np.where(first_is_a, b, a)
    y = df["chose_left"].values
    chose_first = np.where(first_is_a, y, 1 - y)
    p = pd.Series(chose_first).groupby(key).mean()
    return float(np.abs(p.values - 0.5).mean())
