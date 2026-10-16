# name: item_choice_extremity
# description: Mean over distinct pairs of the squared deviation of the pair's choice rate from 0.5 (pair-level consensus strength); observed above null means people agree on items more strongly than the model predicts (model too noisy / misranks strong items), below means the model predicts too much consensus.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str)
    b = df["sequence_b"].to_numpy().astype(str)
    a_first = a <= b
    y = df["chose_left"].to_numpy()
    chose_x = np.where(a_first, y, 1 - y)
    key = pd.Series(np.where(a_first, a, b)) + "|" + pd.Series(np.where(a_first, b, a))
    rate = pd.Series(chose_x).groupby(key.to_numpy()).mean().to_numpy()
    return float(np.mean((rate - 0.5) ** 2))
