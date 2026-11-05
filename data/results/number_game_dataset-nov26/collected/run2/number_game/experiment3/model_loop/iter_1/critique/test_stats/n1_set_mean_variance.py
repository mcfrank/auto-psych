# name: n1_set_mean_variance
# description: Variance across one-example sets of each set's mean yes rate; observed > null means people's breadth of generalization differs more between lone examples than the model's single global doubt/flat-rate mixture allows (observed < null: model over-differentiates lone examples).
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = (s.str.count(",") == 0).values
    r = df["rating"].astype(float).values[one]
    u, inv = np.unique(s.values[one], return_inverse=True)
    m = np.bincount(inv, weights=r) / np.bincount(inv)
    return float(np.var(m))
