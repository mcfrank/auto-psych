# name: long_item_extremity
# description: Mean absolute deviation from 0.5 of the per-pair choice rate (rate of choosing the lexicographically first sequence of the unordered pair) among length-8 pairs; observed above null means the population agrees on length-8 pairs more strongly than the model predicts (model too noisy there), below means the model is overconfident.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    m = (a.str.len() == 8).to_numpy()
    if m.sum() == 0:
        return 0.0
    av = a.to_numpy()[m]; bv = b.to_numpy()[m]; y = df["chose_left"].to_numpy()[m]
    first = np.where(av < bv, av, bv); second = np.where(av < bv, bv, av)
    chose_first = np.where(av == first, y, 1 - y)
    r = pd.Series(chose_first).groupby([first, second]).mean()
    return float(np.mean(np.abs(r.to_numpy() - 0.5)))
