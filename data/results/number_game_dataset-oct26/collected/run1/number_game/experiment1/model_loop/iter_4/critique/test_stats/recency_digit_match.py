# name: recency_digit_match
# description: On 2-4 example sets, yes rate for targets sharing a final digit with the last-shown example but not the first, minus yes rate for targets sharing it with the first but not the last; observed above null means a recency (order) effect that the order-blind average-similarity model cannot produce, below means primacy.
def test_statistic(df):
    s = df["number_set"].astype(str)
    n = s.str.count(",") + 1
    first = s.str.split(",").str[0].astype(int)
    last = s.str.split(",").str[-1].astype(int)
    t = df["target"].astype(int) % 10
    mf = (first % 10) == t
    ml = (last % 10) == t
    r = df["rating"].astype(float)
    a = r[(n > 1) & ml & ~mf]
    b = r[(n > 1) & mf & ~ml]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
