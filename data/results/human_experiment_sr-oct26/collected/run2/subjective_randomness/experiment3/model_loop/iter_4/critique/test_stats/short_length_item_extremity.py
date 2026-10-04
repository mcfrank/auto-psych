# name: short_length_item_extremity
# description: Mean absolute deviation from 0.5 of the per-pair rate of choosing the left-shown sequence (pairs keyed by unordered pair, rate of choosing the lexicographically first sequence) among pairs of length <= 5; observed above null means people are more decisive on short sequences than the model's length-scaled sensitivity predicts, below means less decisive.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    m = (a.str.len() <= 5).to_numpy()
    if m.sum() == 0:
        return 0.0
    a = a[m]; b = b[m]; y = df["chose_left"].to_numpy()[m]
    first = np.where(a.to_numpy() < b.to_numpy(), a.to_numpy(), b.to_numpy())
    second = np.where(a.to_numpy() < b.to_numpy(), b.to_numpy(), a.to_numpy())
    chose_first = np.where(a.to_numpy() == first, y, 1 - y)
    r = pd.Series(chose_first).groupby([first, second]).mean()
    return float(np.mean(np.abs(r.to_numpy() - 0.5)))
