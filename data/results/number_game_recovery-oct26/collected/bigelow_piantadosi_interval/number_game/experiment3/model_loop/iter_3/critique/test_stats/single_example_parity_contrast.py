# name: single_example_parity_contrast
# description: For single-example sets, yes rate on non-example targets with the same parity as the example minus opposite-parity targets; observed above null_mean means people generalize a lone example along odd/even more than the model predicts, below means the model over-weights parity rules for one example.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    e = pd.to_numeric(s.where(single, "0")).astype(int)
    t = df["target"].astype(int)
    keep = single & (t != e)
    same = keep & ((t % 2) == (e % 2))
    diff = keep & ((t % 2) != (e % 2))
    r = df["rating"].astype(float)
    if same.sum() == 0 or diff.sum() == 0:
        return 0.0
    return float(r[same].mean() - r[diff].mean())
