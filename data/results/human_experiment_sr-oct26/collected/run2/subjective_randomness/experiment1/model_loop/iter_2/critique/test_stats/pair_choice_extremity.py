# name: pair_choice_extremity
# description: Mean over unordered pairs of |P(choose lexicographically-first sequence) - 0.5|; observed above null_mean means people agree on pairs more strongly than the model predicts (model too noisy), below means the model is too deterministic.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(dtype=object); b = df["sequence_b"].to_numpy(dtype=object)
    y = df["chose_left"].to_numpy(dtype=float)
    first_left = a <= b
    key = np.where(first_left, a + "|" + b, b + "|" + a)
    c = np.where(first_left, y, 1 - y)
    p = pd.Series(c).groupby(key).mean().to_numpy()
    return float(np.mean(np.abs(p - 0.5)))
