# name: mid_vs_far_tail_drop
# description: Over all sets, yes rate for non-example targets 21-35 from the nearest example minus yes rate for targets 36-70 away; observed > null means generalisation drops between moderate and far distances more than the model's Gaussian-times-gamma falloff predicts, observed < null that the observed tail is flatter (a far-distance plateau the model's Gaussian kills).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).fillna(0).astype(int).values
    has = ex > 0
    t = df["target"].astype(int).values
    dist = np.where(has, np.abs(ex - t[:, None]), 999).min(1)
    r = df["rating"].values.astype(float)
    a = (dist >= 21) & (dist <= 35)
    b = (dist >= 36) & (dist <= 70)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
