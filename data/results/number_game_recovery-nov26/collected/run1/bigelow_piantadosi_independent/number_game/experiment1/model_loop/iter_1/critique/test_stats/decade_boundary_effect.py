# name: decade_boundary_effect
# description: Among non-example targets 1-9 away from the nearest example, yes rate when the target shares its tens digit (decade) with some example minus when it does not; the model's intervals are decade-blind (~0 expected), so observed above null means people generalise within decades more than the model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)  # (rows, <=4), NaN pad
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    dist = np.abs(ex - t[:, None])
    nearest = np.nanmin(dist, axis=1)
    nonmember = nearest > 0
    same_dec = np.any(np.floor(ex / 10) == np.floor(t / 10)[:, None], axis=1)
    m = nonmember & (nearest <= 9)
    a, b = r[m & same_dec], r[m & ~same_dec]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
