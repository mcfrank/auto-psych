# name: far_target_size_principle
# description: Yes rate for targets more than 30 away from every example in 3-4-example sets minus the same far-target yes rate in 1-example sets; observed < model means people tighten generalization with more examples (size principle) more than the averaged exemplar similarity predicts, observed > model means they broaden it more.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    ex = s.str.split(",", expand=True).reindex(columns=range(4))
    E = ex.apply(pd.to_numeric, errors="coerce").values.astype(float)
    k = np.sum(~np.isnan(E), axis=1)
    dist = np.abs(E - t[:, None])
    mind = np.nanmin(dist, axis=1)
    far = mind > 30
    a = far & (k >= 3)
    b = far & (k == 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
