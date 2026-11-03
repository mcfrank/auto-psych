# name: within_set_target_discrimination
# description: Trial-weighted mean, over number_sets with >=40 trials, of the variance across targets of the per-target yes-rate; observed below null means the model discriminates between targets of the same set more sharply than people (people's generalisation flatter), above null means people discriminate targets more sharply than the model.
def test_statistic(df):
    s = df["number_set"].astype(str)
    pt_ = df.groupby([s, df["target"].astype(int)])["rating"].mean()
    v = pt_.groupby(level=0).var(ddof=0)
    n = s.value_counts()
    keep = n[n >= 40].index.intersection(v.index)
    if len(keep) == 0:
        return 0.0
    w = n[keep].to_numpy().astype(float)
    return float(np.sum(v[keep].to_numpy() * w) / w.sum())
