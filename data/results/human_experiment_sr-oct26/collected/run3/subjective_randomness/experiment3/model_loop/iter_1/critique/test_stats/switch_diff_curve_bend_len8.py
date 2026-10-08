# name: switch_diff_curve_bend_len8
# description: At length 8, rate of choosing the more-switching sequence when the switch counts differ by >= 3 minus that rate when they differ by exactly 1; observed above null means choice rises more steeply with switch difference than the model predicts (model too flat for big gaps / too sharp for small), below means it saturates faster than the model.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    sa = (a.str.count("HT") + a.str.count("TH")).values; sb = (b.str.count("HT") + b.str.count("TH")).values
    n = a.str.len().values
    d = np.abs(sa - sb)
    chose_more = np.where(sa > sb, df["chose_left"].values, 1 - df["chose_left"].values)
    big = (n == 8) & (d >= 3); small = (n == 8) & (d == 1)
    if big.sum() == 0 or small.sum() == 0:
        return 0.0
    return float(chose_more[big].mean() - chose_more[small].mean())
