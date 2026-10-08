# name: majority_agreement_length_gap
# description: Agreement with each pair's data-majority choice on length-8 pairs minus agreement on shorter (<8) pairs; observed differing from null_mean means the model mis-scales its score differences across sequence lengths (positive excess: people are relatively more decisive on long sequences than the model predicts).
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    first_is_a = a < b
    key = np.where(first_is_a, a, b) + "|" + np.where(first_is_a, b, a)
    y = df["chose_left"].values
    cf = np.where(first_is_a, y, 1 - y)
    s = pd.Series(cf)
    pm_ = s.groupby(key).transform("mean").values
    maj = (pm_ >= 0.5).astype(float)
    agree = (cf == maj).astype(float)
    L = df["sequence_a"].str.len().values
    long_ = L >= 8
    if long_.sum() == 0 or (~long_).sum() == 0:
        return 0.0
    return float(agree[long_].mean() - agree[~long_].mean())
