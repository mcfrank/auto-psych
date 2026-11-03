# name: decade_share_effect_near
# description: Among non-example targets 1-10 away from the nearest example, yes rate when the target shares a tens digit (decade) with some example minus when it does not; observed above null means people give extra yes for a shared decade that the model lacks, below means the model over-predicts decade sharing.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).values
    t = df["target"].astype(float).values
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    dec = (np.floor(ex / 10) == np.floor(t / 10)[:, None]).any(axis=1)
    near = (dist >= 1) & (dist <= 10)
    r = df["rating"].values.astype(float)
    a, b = near & dec, near & ~dec
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
