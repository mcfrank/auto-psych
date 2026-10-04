# name: pair_choice_extremity
# description: Mean over distinct (unordered) pairs of |share choosing the lexicographically smaller sequence - 0.5|; observed > null means the model's pair predictions are too weak (humans more unanimous), observed < null means the model is overconfident.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    first = np.where(a < b, a, b); second = np.where(a < b, b, a)
    chose_first = np.where(a < b, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    key = pd.Series(first) + "|" + pd.Series(second)
    r = pd.Series(chose_first).groupby(key.to_numpy()).mean()
    return float(np.mean(np.abs(r.to_numpy() - 0.5)))
