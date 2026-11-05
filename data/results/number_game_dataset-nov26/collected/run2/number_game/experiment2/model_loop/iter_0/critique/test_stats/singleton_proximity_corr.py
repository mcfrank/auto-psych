# name: singleton_proximity_corr
# description: Among single-example sets (target != example), Pearson correlation of yes with log(1+|target-example|); observed more negative than null means people use numeric proximity more than the model's one-sided magnitude bounds predict (model under-produces the distance gradient).
def test_statistic(df):
    k = df["number_set"].astype(str).str.count(",") + 1
    sub = df[k == 1]
    x = sub["number_set"].astype(int).values
    t = sub["target"].astype(int).values
    keep = t != x
    d = np.log1p(np.abs(t - x)[keep].astype(float))
    r = sub["rating"].values[keep].astype(float)
    if len(r) < 3 or d.std() == 0 or r.std() == 0:
        return 0.0
    return float(np.corrcoef(d, r)[0, 1])
