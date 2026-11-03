# name: near_vs_far_offdigit_gradient
# description: In 2-4 example sets, among non-example targets sharing no final digit with any example, yes-rate for targets within 5 of the nearest example minus yes-rate for targets more than 20 from every example; observed above the null means people's magnitude-similarity generalization is steeper/stronger than the model's intervals and neighbourhoods produce.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    dig = np.any(np.mod(E, 10) == np.mod(t, 10)[:, None], axis=1)
    base = (n >= 2) & ~dig & (dist > 0)
    a = base & (dist <= 5)
    b = base & (dist > 20)
    r = df["rating"].to_numpy().astype(float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
