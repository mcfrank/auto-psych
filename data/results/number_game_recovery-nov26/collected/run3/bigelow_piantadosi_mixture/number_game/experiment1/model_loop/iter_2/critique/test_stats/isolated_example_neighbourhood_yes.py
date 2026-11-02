# name: isolated_example_neighbourhood_yes
# description: Among 3-4 example sets, yes-rate for non-example targets within 6 of the most isolated example (largest nearest-neighbour gap, >=15) minus yes-rate for targets within 6 of the other examples; observed below the null means people discount an outlying example (treat it as noise) more than the inferred noise rate allows, above means they generalize around it more.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(int).values.astype(float)
    d = np.abs(ex[:, :, None] - ex[:, None, :])
    k = ex.shape[1]
    d[:, np.arange(k), np.arange(k)] = np.inf
    d = np.where(np.isnan(d), np.inf, d)
    nn = d.min(axis=2)
    nn = np.where(np.isnan(ex), -1, nn)
    iso = nn.argmax(axis=1)
    gap = nn.max(axis=1)
    rows = np.arange(len(t))
    iso_val = ex[rows, iso]
    dist = np.abs(ex - t[:, None])
    is_ex = np.nansum(dist == 0, axis=1) > 0
    near_iso = np.abs(t - iso_val) <= 6
    other = dist.copy()
    other[rows, iso] = np.inf
    other = np.where(np.isnan(other), np.inf, other)
    near_other = other.min(axis=1) <= 6
    base = (n >= 3) & (gap >= 15) & (~is_ex)
    a = base & near_iso & ~near_other
    b = base & near_other & ~near_iso
    r = df["rating"].astype(float).values
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
