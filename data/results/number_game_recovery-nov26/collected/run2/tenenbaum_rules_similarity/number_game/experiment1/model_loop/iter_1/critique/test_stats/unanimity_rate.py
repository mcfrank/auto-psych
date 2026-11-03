# name: unanimity_rate
# description: Fraction of (number_set, target) pairs whose repeated responses are all-yes or all-no; observed > null means people agree more (model posteriors too graded/intermediate), observed < null means people are noisier than the model's near-0/1 posteriors.
def test_statistic(df):
    m = df.groupby(["number_set", "target"])["rating"].mean().to_numpy()
    return float(np.mean((m == 0) | (m == 1)))
