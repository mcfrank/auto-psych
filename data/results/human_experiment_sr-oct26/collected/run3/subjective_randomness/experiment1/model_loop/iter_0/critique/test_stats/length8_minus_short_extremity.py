# name: length8_minus_short_extremity
# description: Mean item-level |P(choose left) - 0.5| for length-8 pairs minus that for pairs of length <= 7; positive-vs-null means discrimination grows with length more than the model's length-independent beta implies (negative: less).
def test_statistic(df):
    key = df["sequence_a"] + "|" + df["sequence_b"]
    p = df["chose_left"].groupby(key).mean()
    L = df.groupby(key)["sequence_a"].first().str.len()
    e = (p - 0.5).abs()
    long_ = e[L == 8]; short = e[L < 8]
    if len(long_) == 0 or len(short) == 0:
        return 0.0
    return float(long_.mean() - short.mean())
