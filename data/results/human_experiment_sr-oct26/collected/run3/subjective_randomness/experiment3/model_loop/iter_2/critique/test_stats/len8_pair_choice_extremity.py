# name: len8_pair_choice_extremity
# description: Mean over length-8 pairs of |pair choice rate - 0.5| (canonical orientation, pooled over participants); observed above null means the population consensus on long pairs is sharper than the model's pooled predictions, below means the model predicts more extreme consensus than people show.
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    n = df["sequence_a"].str.len().values
    first = np.where(a <= b, a, b); second = np.where(a <= b, b, a)
    pick_first = np.where(a <= b, df["chose_left"].values, 1 - df["chose_left"].values)
    m = n == 8
    if not m.any():
        return 0.0
    key = (pd.Series(first[m]) + "|" + pd.Series(second[m])).values
    rate = pd.Series(pick_first[m]).groupby(key).mean().values
    return float(np.abs(rate - 0.5).mean())
