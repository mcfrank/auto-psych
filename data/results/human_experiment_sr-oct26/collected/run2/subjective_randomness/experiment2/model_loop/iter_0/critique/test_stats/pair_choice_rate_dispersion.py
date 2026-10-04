# name: pair_choice_rate_dispersion
# description: Mean over unordered stimulus pairs of |rate of choosing the lexicographically-first sequence - 0.5|; observed above null means people are more consensual (extreme) on individual pairs than the model predicts, below means the model is overconfident.
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    first_left = a < b
    key = np.where(first_left, a + "|" + b, b + "|" + a)
    chose_first = np.where(first_left, df["chose_left"], 1 - df["chose_left"])
    r = pd.Series(chose_first).groupby(key).mean()
    return float((r - 0.5).abs().mean())
