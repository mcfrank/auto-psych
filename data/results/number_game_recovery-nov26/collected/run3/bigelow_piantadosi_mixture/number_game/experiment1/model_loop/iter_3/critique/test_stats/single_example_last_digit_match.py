# name: single_example_last_digit_match
# description: For one-example sets, yes rate for targets more than 10 from the example that share its last digit minus those that do not; observed > null means people generalise a lone example along its last digit more than the model's "ends in d" rules produce, observed < null less.
def test_statistic(df):
    one = ~df["number_set"].astype(str).str.contains(",")
    d = df[one]
    if len(d) == 0:
        return 0.0
    e = d["number_set"].astype(int).values
    t = d["target"].astype(int).values
    r = d["rating"].astype(float).values
    far = np.abs(t - e) > 10
    same = (t % 10) == (e % 10)
    a, b = far & same, far & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
