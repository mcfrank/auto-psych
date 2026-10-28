# name: single_example_shared_digit_contrast
# description: Among single-example sets, yes rate to non-example targets sharing a digit (tens or ones, as a digit string) with the example minus yes rate to targets sharing no digit; observed above the null means people generalize from one number by digit similarity more than the model predicts, below means the model over-weights digit rules.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",").to_numpy()
    t = df["target"].astype(int)
    ts = t.astype(str).to_numpy()
    sv = s.to_numpy()
    share = np.array([bool(set(a) & set(b)) for a, b in zip(ts, sv)])
    isex = ts == sv
    r = df["rating"].to_numpy()
    ms, mn = single & share & ~isex, single & ~share
    if ms.sum() == 0 or mn.sum() == 0:
        return 0.0
    return float(r[ms].mean() - r[mn].mean())
