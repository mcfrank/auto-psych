# name: multi_item_mean_variance
# description: Over multi-example (2-4) sets only, variance across (set, target) pairs of the pair's mean yes rate; observed > null means people separate endorsed from rejected targets more sharply than the model's overlap judgment (model too graded/under-confident on multi-example sets), observed < null the reverse.
def test_statistic(df):
    s = df["number_set"].astype(str)
    multi = (s.str.count(",") > 0).values
    key = (s.values[multi] + "|" + df["target"].astype(str).values[multi])
    r = df["rating"].astype(float).values[multi]
    u, inv = np.unique(key, return_inverse=True)
    m = np.bincount(inv, weights=r) / np.bincount(inv)
    return float(np.var(m))
