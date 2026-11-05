# name: single_example_proximity_gap
# description: In single-example sets, yes-rate on targets within 15 of the example minus yes-rate on targets more than 40 away; the model has no numeric-distance term for n=1 (only one-sided bounds), so observed above null means people generalize from a single example by magnitude similarity the model misses, below means they ignore magnitude more than modelled.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    dist = np.abs(E[:, 0] - t)
    ok = (n == 1) & (dist > 0)
    a = ok & (dist <= 15)
    b = ok & (dist > 40)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
