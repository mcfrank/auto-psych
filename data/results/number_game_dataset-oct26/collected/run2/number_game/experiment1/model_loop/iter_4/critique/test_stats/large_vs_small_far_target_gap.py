# name: large_vs_small_far_target_gap
# description: Among targets more than 20 away from every example, yes rate for targets > 50 minus yes rate for targets <= 50; observed != model means people carry a target-magnitude bias the similarity model lacks (observed > model: they over-endorse large far targets; observed < model: they over-endorse small far targets).
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    ex = s.str.split(",", expand=True).reindex(columns=range(4))
    E = ex.apply(pd.to_numeric, errors="coerce").values.astype(float)
    mind = np.nanmin(np.abs(E - t[:, None]), axis=1)
    far = mind > 20
    a = far & (t > 50)
    b = far & (t <= 50)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
